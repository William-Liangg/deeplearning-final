"""Test whether LEACE-erased opponent style affects opponent-action prediction."""

import argparse
from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
import sys

from concept_erasure import LeaceEraser
import numpy as np
import torch

from opponent_style_probing.hand20_style_probe import (
    checkpoint_identity,
    make_session_partition,
    read_vocabulary,
)
from opponent_style_probing.leace_style_transformation import (
    distortion_measurements,
    fit_leace,
    fresh_probe_measurements,
    maximum_class_mean_separation,
    random_rank_eraser,
    transformation_rank,
    transform_with_leace,
)
from opponent_style_probing.opponent_action_prediction import (
    BLOCK4_RESIDUAL_HOOK,
    OPPONENT_ACTIONS,
    ActionPredictions,
    action_predictions,
    bootstrap_mean_difference,
    extract_predictor_states_and_logits,
    find_next_opponent_targets,
    prediction_measurements,
)
from training.train import load_checkpoint


SPLIT_SEED = 0
SHUFFLE_SEED = 1
RANDOM_CONTROL_SEEDS = tuple(range(2, 12))
BOOTSTRAP_SEED = 20
BOOTSTRAP_REPEATS = 1000
MATCHING_TOLERANCE = 1e-10
NUMERICAL_TOLERANCE = 1e-8
HOOK_TOLERANCE = 1e-4
MAXIMUM_HELD_OUT_ACCURACY = 0.30
CHECKPOINT_PATH = Path("training/checkpoints/seed0/step_008439.pt")
TOKEN_PATH = Path("data/full/val_tokens.npy")
STYLE_PATH = Path("data/full/val_styles.npy")
VOCABULARY_PATH = Path("data/full/vocab.json")
MEASUREMENT_PATH = Path("opponent_style_probing/measurements/leace_opponent_action_seed0.json")


@dataclass(frozen=True)
class ResidualStateIntervention:
    center: np.ndarray
    coordinate_encoder: np.ndarray
    state_decoder: np.ndarray
    displacement_scale: float


@dataclass(frozen=True)
class InterventionOutcome:
    predictions: ActionPredictions
    measurements: dict[str, object]


def intervention_from_leace(
    eraser: LeaceEraser,
    displacement_scale: float = 1.0,
) -> ResidualStateIntervention:
    if eraser.bias is None:
        raise ValueError("The approved intervention requires affine LEACE")
    return ResidualStateIntervention(
        center=eraser.bias.detach().cpu().numpy(),
        coordinate_encoder=eraser.proj_right.mH.detach().cpu().numpy(),
        state_decoder=eraser.proj_left.mH.detach().cpu().numpy(),
        displacement_scale=displacement_scale,
    )


def intervention_from_random_projection(
    fitting_mean: np.ndarray,
    residual_width: int,
    displacement_rank: int,
    random_seed: int,
    displacement_scale: float = 1.0,
) -> ResidualStateIntervention:
    eraser = random_rank_eraser(
        fitting_mean,
        residual_width,
        displacement_rank,
        random_seed,
    )
    return ResidualStateIntervention(
        center=eraser.center,
        coordinate_encoder=eraser.directions.T,
        state_decoder=eraser.directions,
        displacement_scale=displacement_scale,
    )


def transform_states(
    state_values: np.ndarray,
    intervention: ResidualStateIntervention,
) -> np.ndarray:
    centered_values = state_values - intervention.center
    displacements = (
        centered_values @ intervention.coordinate_encoder @ intervention.state_decoder
    )
    return state_values - intervention.displacement_scale * displacements


def edit_selected_states(
    residual_states: torch.Tensor,
    predictor_positions: torch.Tensor,
    intervention: ResidualStateIntervention,
) -> torch.Tensor:
    """Transform one selected state per row and leave all other states unchanged."""

    edited_states = residual_states.clone()
    batch_rows = torch.arange(len(predictor_positions), device=residual_states.device)
    selected_states = edited_states[batch_rows, predictor_positions]
    center = torch.as_tensor(
        intervention.center,
        dtype=residual_states.dtype,
        device=residual_states.device,
    )
    coordinate_encoder = torch.as_tensor(
        intervention.coordinate_encoder,
        dtype=residual_states.dtype,
        device=residual_states.device,
    )
    state_decoder = torch.as_tensor(
        intervention.state_decoder,
        dtype=residual_states.dtype,
        device=residual_states.device,
    )
    displacements = (selected_states - center) @ coordinate_encoder @ state_decoder
    edited_states[batch_rows, predictor_positions] = (
        selected_states - intervention.displacement_scale * displacements
    )
    return edited_states


def displacement_rank(intervention: ResidualStateIntervention) -> int:
    displacement_matrix = intervention.coordinate_encoder @ intervention.state_decoder
    return int(np.linalg.matrix_rank(displacement_matrix))


def mean_squared_change(
    original_values: np.ndarray,
    transformed_values: np.ndarray,
) -> float:
    if original_values.shape != transformed_values.shape:
        raise ValueError("Activation arrays must have matching shapes")
    return float(np.square(transformed_values - original_values).mean())


def centered_energy_fraction_change(
    original_values: np.ndarray,
    transformed_values: np.ndarray,
    fitting_mean: np.ndarray,
) -> float:
    original_energy = np.square(original_values - fitting_mean).sum()
    transformed_energy = np.square(transformed_values - fitting_mean).sum()
    if original_energy <= 0:
        raise ValueError("Centered activation energy must be positive")
    return float((transformed_energy - original_energy) / original_energy)


def magnitude_matching_scale(
    fitting_values: np.ndarray,
    reference_fitting_values: np.ndarray,
    control_fitting_values: np.ndarray,
) -> float:
    reference_change = mean_squared_change(fitting_values, reference_fitting_values)
    control_change = mean_squared_change(fitting_values, control_fitting_values)
    if reference_change <= 0 or control_change <= 0:
        raise ValueError("Magnitude matching requires positive fitting distortion")
    return float(np.sqrt(reference_change / control_change))


def with_displacement_scale(
    intervention: ResidualStateIntervention,
    displacement_scale: float,
) -> ResidualStateIntervention:
    if not np.isfinite(displacement_scale) or displacement_scale <= 0:
        raise ValueError("Displacement scale must be positive and finite")
    return ResidualStateIntervention(
        center=intervention.center,
        coordinate_encoder=intervention.coordinate_encoder,
        state_decoder=intervention.state_decoder,
        displacement_scale=displacement_scale,
    )


def run_intervention_logits(
    model,
    token_rows: np.ndarray,
    predictor_positions: np.ndarray,
    intervention: ResidualStateIntervention,
    batch_size: int,
) -> np.ndarray:
    """Run a fixed intervention without accepting style labels."""

    predictor_logits = np.empty((len(token_rows), int(model.cfg.d_vocab)), dtype=np.float32)
    model_device = next(model.parameters()).device
    with torch.inference_mode():
        for batch_start in range(0, len(token_rows), batch_size):
            batch_stop = min(batch_start + batch_size, len(token_rows))
            batch_positions = predictor_positions[batch_start:batch_stop]
            batch_length = int(batch_positions.max()) + 1
            batch_tokens = torch.from_numpy(
                np.asarray(token_rows[batch_start:batch_stop, :batch_length], dtype=np.int64)
            ).to(model_device)
            position_indices = torch.from_numpy(batch_positions.astype(np.int64)).to(model_device)

            def intervention_hook(residual_states: torch.Tensor, hook) -> torch.Tensor:
                return edit_selected_states(residual_states, position_indices, intervention)

            logits = model.run_with_hooks(
                batch_tokens,
                return_type="logits",
                fwd_hooks=[(BLOCK4_RESIDUAL_HOOK, intervention_hook)],
            )
            batch_rows = torch.arange(batch_stop - batch_start, device=model_device)
            predictor_logits[batch_start:batch_stop] = (
                logits[batch_rows, position_indices].float().cpu().numpy()
            )
    if not np.isfinite(predictor_logits).all():
        raise ValueError("Intervention logits must be finite")
    return predictor_logits


def intervention_measurements(
    original_states: np.ndarray,
    transformed_states: np.ndarray,
    style_labels: np.ndarray,
    fitting_indices: np.ndarray,
    test_indices: np.ndarray,
    class_ids: np.ndarray,
    fitting_mean: np.ndarray,
    intervention: ResidualStateIntervention,
) -> dict[str, object]:
    if not np.isfinite(transformed_states).all():
        raise ValueError("Transformed states must be finite")
    fitting_distortion = mean_squared_change(
        original_states[fitting_indices],
        transformed_states[fitting_indices],
    )
    held_out_distortion = distortion_measurements(
        original_states[test_indices],
        transformed_states[test_indices],
        fitting_mean,
        displacement_rank(intervention),
    )
    return {
        "displacement_scale": intervention.displacement_scale,
        "fitting_mean_squared_activation_change": fitting_distortion,
        "held_out_centered_energy_fraction_change": centered_energy_fraction_change(
            original_states[test_indices],
            transformed_states[test_indices],
            fitting_mean,
        ),
        **fresh_probe_measurements(
            transformed_states,
            style_labels,
            fitting_indices,
            test_indices,
            class_ids,
        ),
        **held_out_distortion,
    }


def evaluate_intervention(
    model,
    test_token_rows: np.ndarray,
    test_positions: np.ndarray,
    test_targets: np.ndarray,
    opponent_action_ids: tuple[int, ...],
    intervention: ResidualStateIntervention,
    untouched_predictions: ActionPredictions,
    state_measurements: dict[str, object],
    batch_size: int,
    bootstrap_seed: int,
) -> InterventionOutcome:
    logits = run_intervention_logits(
        model,
        test_token_rows,
        test_positions,
        intervention,
        batch_size,
    )
    predictions = action_predictions(logits, test_targets, opponent_action_ids)
    if (
        not np.isfinite(predictions.losses).all()
        or not np.isfinite(predictions.correct_probabilities).all()
    ):
        raise ValueError("Every per-session prediction measurement must be finite")
    measurements = {
        **state_measurements,
        **prediction_measurements(predictions),
        "loss_increase_over_untouched": bootstrap_mean_difference(
            predictions.losses,
            untouched_predictions.losses,
            BOOTSTRAP_REPEATS,
            bootstrap_seed,
        ),
    }
    return InterventionOutcome(predictions=predictions, measurements=measurements)


def comparison_measurement(
    style_losses: np.ndarray,
    control_losses: np.ndarray,
    bootstrap_seed: int,
) -> dict[str, float | int]:
    return bootstrap_mean_difference(
        style_losses,
        control_losses,
        BOOTSTRAP_REPEATS,
        bootstrap_seed,
    )


def all_measurement_numbers_finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(all_measurement_numbers_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_measurement_numbers_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def run_leace_opponent_action_intervention(batch_size: int, device: str) -> dict[str, object]:
    vocabulary, style_names = read_vocabulary(VOCABULARY_PATH)
    token_rows = np.load(TOKEN_PATH, mmap_mode="r")
    style_labels = np.load(STYLE_PATH)
    class_ids = np.arange(len(style_names), dtype=np.int64)
    partition = make_session_partition(style_labels, SPLIT_SEED)
    fitting_indices = partition.fitting_indices
    test_indices = partition.test_indices
    fitting_labels = style_labels[fitting_indices]
    opponent_action_ids = tuple(vocabulary.index(action_name) for action_name in OPPONENT_ACTIONS)
    predictor_positions, target_ids = find_next_opponent_targets(
        token_rows,
        vocabulary.index("[HAND]"),
        opponent_action_ids,
    )

    model, checkpoint_state = load_checkpoint(str(CHECKPOINT_PATH), device=device)
    predictor_states, untouched_logits = extract_predictor_states_and_logits(
        model,
        token_rows,
        predictor_positions,
        batch_size,
    )
    if predictor_states.shape != (len(token_rows), 256):
        raise ValueError("The approved study requires one width-256 state per session")
    fitting_mean = predictor_states[fitting_indices].mean(axis=0, dtype=np.float64)

    style_eraser = fit_leace(
        predictor_states,
        fitting_indices,
        fitting_labels,
        len(style_names),
    )
    style_intervention = intervention_from_leace(style_eraser)
    style_states = transform_states(predictor_states, style_intervention)
    style_fitting_states = style_states[fitting_indices]
    style_rank = transformation_rank(style_eraser)
    if displacement_rank(style_intervention) != style_rank:
        raise ValueError("LEACE displacement ranks disagree")

    repeated_eraser = fit_leace(
        predictor_states,
        fitting_indices,
        fitting_labels,
        len(style_names),
    )
    repeated_states = transform_with_leace(predictor_states, repeated_eraser)
    deterministic_difference = float(np.max(np.abs(style_states - repeated_states)))
    fitting_tensor = torch.from_numpy(
        np.asarray(predictor_states[fitting_indices], dtype=np.float64)
    )
    once_erased = style_eraser(fitting_tensor)
    twice_erased = style_eraser(once_erased)
    idempotence_difference = float(torch.max(torch.abs(once_erased - twice_erased)).item())
    class_mean_separation_before = maximum_class_mean_separation(
        predictor_states,
        fitting_indices,
        fitting_labels,
        len(style_names),
    )
    class_mean_separation_after = maximum_class_mean_separation(
        style_states,
        fitting_indices,
        fitting_labels,
        len(style_names),
    )
    if deterministic_difference > NUMERICAL_TOLERANCE:
        raise ValueError("Repeated LEACE fitting must be deterministic")
    if idempotence_difference > NUMERICAL_TOLERANCE:
        raise ValueError("LEACE must be idempotent within numerical tolerance")
    if class_mean_separation_after > NUMERICAL_TOLERANCE:
        raise ValueError("LEACE must equalize fitting class means")

    shuffled_labels = np.random.default_rng(SHUFFLE_SEED).permutation(fitting_labels)
    shuffled_eraser = fit_leace(
        predictor_states,
        fitting_indices,
        shuffled_labels,
        len(style_names),
    )
    shuffled_intervention = intervention_from_leace(shuffled_eraser)
    shuffled_states = transform_states(predictor_states, shuffled_intervention)
    shuffled_scale = magnitude_matching_scale(
        predictor_states[fitting_indices],
        style_fitting_states,
        shuffled_states[fitting_indices],
    )
    matched_shuffled_intervention = with_displacement_scale(
        shuffled_intervention,
        shuffled_scale,
    )

    test_token_rows = np.asarray(token_rows[test_indices])
    test_positions = predictor_positions[test_indices]
    test_targets = target_ids[test_indices]
    untouched_predictions = action_predictions(
        untouched_logits[test_indices],
        test_targets,
        opponent_action_ids,
    )
    untouched_state_measurements = {
        "displacement_scale": 0.0,
        "fitting_mean_squared_activation_change": 0.0,
        "held_out_centered_energy_fraction_change": 0.0,
        **fresh_probe_measurements(
            predictor_states,
            style_labels,
            fitting_indices,
            test_indices,
            class_ids,
        ),
        **distortion_measurements(
            predictor_states[test_indices],
            predictor_states[test_indices],
            fitting_mean,
            0,
        ),
    }
    untouched_measurements = {
        **untouched_state_measurements,
        **prediction_measurements(untouched_predictions),
    }

    style_state_measurements = intervention_measurements(
        predictor_states,
        style_states,
        style_labels,
        fitting_indices,
        test_indices,
        class_ids,
        fitting_mean,
        style_intervention,
    )
    style_outcome = evaluate_intervention(
        model,
        test_token_rows,
        test_positions,
        test_targets,
        opponent_action_ids,
        style_intervention,
        untouched_predictions,
        style_state_measurements,
        batch_size,
        BOOTSTRAP_SEED,
    )

    shuffled_state_measurements = intervention_measurements(
        predictor_states,
        shuffled_states,
        style_labels,
        fitting_indices,
        test_indices,
        class_ids,
        fitting_mean,
        shuffled_intervention,
    )
    shuffled_outcome = evaluate_intervention(
        model,
        test_token_rows,
        test_positions,
        test_targets,
        opponent_action_ids,
        shuffled_intervention,
        untouched_predictions,
        shuffled_state_measurements,
        batch_size,
        BOOTSTRAP_SEED + 1,
    )

    matched_shuffled_states = transform_states(
        predictor_states,
        matched_shuffled_intervention,
    )
    reference_fitting_distortion = mean_squared_change(
        predictor_states[fitting_indices],
        style_fitting_states,
    )
    matched_shuffled_fitting_distortion = mean_squared_change(
        predictor_states[fitting_indices],
        matched_shuffled_states[fitting_indices],
    )
    if abs(matched_shuffled_fitting_distortion - reference_fitting_distortion) > MATCHING_TOLERANCE:
        raise ValueError("Shuffled displacement does not match fitting distortion")
    matched_shuffled_state_measurements = intervention_measurements(
        predictor_states,
        matched_shuffled_states,
        style_labels,
        fitting_indices,
        test_indices,
        class_ids,
        fitting_mean,
        matched_shuffled_intervention,
    )
    matched_shuffled_outcome = evaluate_intervention(
        model,
        test_token_rows,
        test_positions,
        test_targets,
        opponent_action_ids,
        matched_shuffled_intervention,
        untouched_predictions,
        matched_shuffled_state_measurements,
        batch_size,
        BOOTSTRAP_SEED + 2,
    )

    random_outcomes: list[tuple[int, InterventionOutcome]] = []
    matched_random_outcomes: list[tuple[int, InterventionOutcome]] = []
    for control_index, random_seed in enumerate(RANDOM_CONTROL_SEEDS):
        random_intervention = intervention_from_random_projection(
            fitting_mean,
            predictor_states.shape[1],
            style_rank,
            random_seed,
        )
        random_states = transform_states(predictor_states, random_intervention)
        random_scale = magnitude_matching_scale(
            predictor_states[fitting_indices],
            style_fitting_states,
            random_states[fitting_indices],
        )
        matched_random_intervention = with_displacement_scale(
            random_intervention,
            random_scale,
        )
        matched_random_states = transform_states(
            predictor_states,
            matched_random_intervention,
        )
        matched_random_fitting_distortion = mean_squared_change(
            predictor_states[fitting_indices],
            matched_random_states[fitting_indices],
        )
        if abs(matched_random_fitting_distortion - reference_fitting_distortion) > MATCHING_TOLERANCE:
            raise ValueError(f"Random seed {random_seed} does not match fitting distortion")
        if displacement_rank(random_intervention) != style_rank:
            raise ValueError(f"Random seed {random_seed} does not match displacement rank")
        random_outcome = evaluate_intervention(
            model,
            test_token_rows,
            test_positions,
            test_targets,
            opponent_action_ids,
            random_intervention,
            untouched_predictions,
            intervention_measurements(
                predictor_states,
                random_states,
                style_labels,
                fitting_indices,
                test_indices,
                class_ids,
                fitting_mean,
                random_intervention,
            ),
            batch_size,
            BOOTSTRAP_SEED + 10 + control_index,
        )
        matched_random_outcome = evaluate_intervention(
            model,
            test_token_rows,
            test_positions,
            test_targets,
            opponent_action_ids,
            matched_random_intervention,
            untouched_predictions,
            intervention_measurements(
                predictor_states,
                matched_random_states,
                style_labels,
                fitting_indices,
                test_indices,
                class_ids,
                fitting_mean,
                matched_random_intervention,
            ),
            batch_size,
            BOOTSTRAP_SEED + 30 + control_index,
        )
        random_outcomes.append((random_seed, random_outcome))
        matched_random_outcomes.append((random_seed, matched_random_outcome))

    ordinary_comparisons = [
        {
            "condition": "shuffled_label_leace",
            **comparison_measurement(
                style_outcome.predictions.losses,
                shuffled_outcome.predictions.losses,
                BOOTSTRAP_SEED + 50,
            ),
        }
    ]
    ordinary_comparisons.extend(
        {
            "condition": f"random_rank_three_seed{random_seed}",
            **comparison_measurement(
                style_outcome.predictions.losses,
                outcome.predictions.losses,
                BOOTSTRAP_SEED + 51 + comparison_index,
            ),
        }
        for comparison_index, (random_seed, outcome) in enumerate(random_outcomes)
    )
    magnitude_comparisons = [
        {
            "condition": "magnitude_matched_shuffled_displacement",
            **comparison_measurement(
                style_outcome.predictions.losses,
                matched_shuffled_outcome.predictions.losses,
                BOOTSTRAP_SEED + 70,
            ),
        }
    ]
    magnitude_comparisons.extend(
        {
            "condition": f"magnitude_matched_random_seed{random_seed}",
            **comparison_measurement(
                style_outcome.predictions.losses,
                outcome.predictions.losses,
                BOOTSTRAP_SEED + 71 + comparison_index,
            ),
        }
        for comparison_index, (random_seed, outcome) in enumerate(matched_random_outcomes)
    )

    hook_check_count = min(batch_size, len(test_indices))
    hook_check_states = torch.from_numpy(
        np.asarray(predictor_states[test_indices[:hook_check_count]], dtype=np.float32)
    ).unsqueeze(1)
    hook_check_positions = torch.zeros(hook_check_count, dtype=torch.int64)
    hook_check_values = edit_selected_states(
        hook_check_states,
        hook_check_positions,
        style_intervention,
    )[:, 0].numpy()
    hook_offline_difference = float(np.max(np.abs(
        hook_check_values - style_states[test_indices[:hook_check_count]]
    )))
    if hook_offline_difference > HOOK_TOLERANCE:
        raise ValueError("Hook-time and offline LEACE transformations disagree")

    style_over_untouched = style_outcome.measurements["loss_increase_over_untouched"]
    decision = {
        "style_balanced_accuracy_at_most_30_percent": bool(
            style_outcome.measurements["balanced_accuracy"] <= MAXIMUM_HELD_OUT_ACCURACY
        ),
        "style_loss_increase_interval_above_zero": bool(
            style_over_untouched["lower_bound"] > 0
        ),
        "style_loss_exceeds_every_ordinary_control": all(
            comparison["lower_bound"] > 0 for comparison in ordinary_comparisons
        ),
        "style_loss_exceeds_every_magnitude_matched_control": all(
            comparison["lower_bound"] > 0 for comparison in magnitude_comparisons
        ),
    }
    decision["style_specific_behavioral_evidence"] = all(decision.values())

    measurements: dict[str, object] = {
        "question": "Does linearly readable opponent style at block 4 contribute to predicting the first opponent action after hand 20?",
        "inputs": {
            "checkpoint": checkpoint_identity(CHECKPOINT_PATH, checkpoint_state),
            "tokens": str(TOKEN_PATH),
            "styles": str(STYLE_PATH),
            "vocabulary": str(VOCABULARY_PATH),
        },
        "experiment": {
            "sessions": len(token_rows),
            "classifier_fitting_sessions": len(fitting_indices),
            "test_sessions": len(test_indices),
            "activation_hook": BLOCK4_RESIDUAL_HOOK,
            "target": "first opponent action after hand 20",
            "split_seed": SPLIT_SEED,
            "shuffle_seed": SHUFFLE_SEED,
            "random_control_seeds": list(RANDOM_CONTROL_SEEDS),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_repetitions": BOOTSTRAP_REPEATS,
            "matching_tolerance": MATCHING_TOLERANCE,
            "hook_tolerance": HOOK_TOLERANCE,
            "batch_size": batch_size,
            "device": device,
        },
        "conditions": {
            "untouched": untouched_measurements,
            "true_label_leace": style_outcome.measurements,
            "shuffled_label_leace": shuffled_outcome.measurements,
            "magnitude_matched_shuffled_displacement": matched_shuffled_outcome.measurements,
            "random_rank_three": [
                {"seed": random_seed, **outcome.measurements}
                for random_seed, outcome in random_outcomes
            ],
            "magnitude_matched_random_rank_three": [
                {"seed": random_seed, **outcome.measurements}
                for random_seed, outcome in matched_random_outcomes
            ],
        },
        "paired_loss_comparisons": {
            "true_leace_minus_ordinary_controls": ordinary_comparisons,
            "true_leace_minus_magnitude_matched_controls": magnitude_comparisons,
        },
        "validation": {
            "maximum_fitting_class_mean_separation_before": class_mean_separation_before,
            "maximum_fitting_class_mean_separation_after": class_mean_separation_after,
            "maximum_deterministic_transform_difference": deterministic_difference,
            "maximum_idempotence_difference": idempotence_difference,
            "maximum_hook_offline_difference": hook_offline_difference,
            "reference_fitting_mean_squared_activation_change": reference_fitting_distortion,
            "numerical_tolerance": NUMERICAL_TOLERANCE,
            "magnitude_matching_tolerance": MATCHING_TOLERANCE,
            "hook_tolerance": HOOK_TOLERANCE,
            "all_values_finite": True,
        },
        "decision": decision,
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformer_lens": version("transformer-lens"),
            "numpy": np.__version__,
            "scikit_learn": version("scikit-learn"),
            "concept_erasure": version("concept-erasure"),
        },
    }
    if not all_measurement_numbers_finite(measurements):
        raise ValueError("Every reported numeric measurement must be finite")
    MEASUREMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEASUREMENT_PATH.write_text(json.dumps(measurements, indent=2), encoding="utf-8")
    return measurements


def main() -> None:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--batch-size", type=int, default=64)
    argument_parser.add_argument("--device", default="cpu")
    arguments = argument_parser.parse_args()
    measurements = run_leace_opponent_action_intervention(
        arguments.batch_size,
        arguments.device,
    )
    print(json.dumps({
        "conditions": measurements["conditions"],
        "paired_loss_comparisons": measurements["paired_loss_comparisons"],
        "decision": measurements["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
