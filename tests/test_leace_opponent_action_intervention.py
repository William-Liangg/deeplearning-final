import inspect
import json
from pathlib import Path

import numpy as np
import torch

from opponent_style_probing.leace_opponent_action_intervention import (
    all_measurement_numbers_finite,
    comparison_measurement,
    displacement_rank,
    edit_selected_states,
    intervention_from_leace,
    intervention_from_random_projection,
    magnitude_matching_scale,
    mean_squared_change,
    run_intervention_logits,
    transform_states,
    with_displacement_scale,
)
from opponent_style_probing.leace_style_transformation import (
    fit_leace,
    fresh_probe_measurements,
    maximum_class_mean_separation,
    one_hot_concepts,
    transformation_rank,
    transform_with_leace,
)
from opponent_style_probing.opponent_action_prediction import (
    OPPONENT_ACTIONS,
    extract_predictor_states_and_logits,
    find_next_opponent_targets,
)
from training.train import load_checkpoint


HAND_TOKEN_ID = 1
HERO_CHECK_ID = 9
OPPONENT_ACTION_IDS = (12, 13, 14, 15)
OUTCOME_ID = 6


def session_tokens(hand_actions: list[list[int]]) -> np.ndarray:
    if len(hand_actions) != 50:
        raise ValueError("A test session requires exactly 50 hands")
    tokens = np.zeros(1024, dtype=np.uint16)
    write_position = 0
    for actions in hand_actions:
        hand_tokens = [HAND_TOKEN_ID, *actions, OUTCOME_ID]
        tokens[write_position : write_position + len(hand_tokens)] = hand_tokens
        write_position += len(hand_tokens)
    return tokens


def test_selects_first_opponent_action_after_twenty_hands() -> None:
    ordinary_hand = [HERO_CHECK_ID, OPPONENT_ACTION_IDS[1]]
    delayed_action_hand = [HERO_CHECK_ID, OPPONENT_ACTION_IDS[3], OPPONENT_ACTION_IDS[0]]
    immediate_action_hand = [OPPONENT_ACTION_IDS[2], HERO_CHECK_ID]
    token_rows = np.stack(
        [
            session_tokens([ordinary_hand] * 20 + [delayed_action_hand] + [ordinary_hand] * 29),
            session_tokens([ordinary_hand] * 20 + [immediate_action_hand] + [ordinary_hand] * 29),
        ]
    )

    predictor_positions, target_ids = find_next_opponent_targets(
        token_rows,
        HAND_TOKEN_ID,
        OPPONENT_ACTION_IDS,
    )

    np.testing.assert_array_equal(target_ids, [OPPONENT_ACTION_IDS[3], OPPONENT_ACTION_IDS[2]])
    np.testing.assert_array_equal(token_rows[np.arange(2), predictor_positions + 1], target_ids)
    assert token_rows[0, predictor_positions[0]] == HERO_CHECK_ID
    assert token_rows[1, predictor_positions[1]] == HAND_TOKEN_ID


def test_rejects_session_without_later_opponent_action() -> None:
    ordinary_hand = [HERO_CHECK_ID, OPPONENT_ACTION_IDS[1]]
    hero_only_hand = [HERO_CHECK_ID]
    token_rows = session_tokens([ordinary_hand] * 20 + [hero_only_hand] * 30)[np.newaxis, :]

    with np.testing.assert_raises_regex(ValueError, "no opponent action after hand 20"):
        find_next_opponent_targets(token_rows, HAND_TOKEN_ID, OPPONENT_ACTION_IDS)


def synthetic_states() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    random_generator = np.random.default_rng(4)
    labels = np.tile(np.arange(4, dtype=np.int64), 50)
    values = random_generator.normal(0, 0.2, (200, 12))
    values[np.arange(200), labels] += 2.0
    fitting_indices = np.arange(140)
    return values, labels, fitting_indices


def test_leace_uses_fitting_labels_and_removes_fresh_linear_readability() -> None:
    values, labels, fitting_indices = synthetic_states()
    fitting_labels = labels[fitting_indices]
    eraser = fit_leace(values, fitting_indices, fitting_labels, class_count=4)
    transformed_values = transform_with_leace(values, eraser)
    test_indices = np.arange(140, 200)

    concepts = one_hot_concepts(fitting_labels, class_count=4)
    measurements = fresh_probe_measurements(
        transformed_values,
        labels,
        fitting_indices,
        test_indices,
        np.arange(4),
    )

    assert concepts.shape == (140, 4)
    assert maximum_class_mean_separation(
        transformed_values,
        fitting_indices,
        fitting_labels,
        class_count=4,
    ) < 1e-8
    assert measurements["balanced_accuracy"] < 0.35


def test_leace_is_deterministic_finite_idempotent_and_rank_three() -> None:
    values, labels, fitting_indices = synthetic_states()
    fitting_labels = labels[fitting_indices]
    first_eraser = fit_leace(values, fitting_indices, fitting_labels, class_count=4)
    second_eraser = fit_leace(values, fitting_indices, fitting_labels, class_count=4)

    first_values = transform_with_leace(values, first_eraser)
    repeated_values = transform_with_leace(values, second_eraser)
    twice_erased = transform_with_leace(first_values, first_eraser)

    assert transformation_rank(first_eraser) == 3
    assert np.isfinite(first_values).all()
    np.testing.assert_allclose(first_values, repeated_values, atol=1e-10)
    np.testing.assert_allclose(first_values, twice_erased, atol=1e-10)


def test_hook_edit_changes_only_selected_rows_and_positions() -> None:
    residual_states = torch.arange(60, dtype=torch.float32).reshape(3, 5, 4)
    original_states = residual_states.clone()
    positions = torch.tensor([0, 2, 4])
    intervention = intervention_from_random_projection(
        np.zeros(4),
        residual_width=4,
        displacement_rank=3,
        random_seed=2,
    )

    edited_states = edit_selected_states(residual_states, positions, intervention)

    selected = torch.zeros((3, 5), dtype=torch.bool)
    selected[torch.arange(3), positions] = True
    torch.testing.assert_close(edited_states[~selected], original_states[~selected])
    assert torch.max(torch.abs(edited_states[selected] - original_states[selected])) > 0
    torch.testing.assert_close(residual_states, original_states)


def test_hook_time_leace_matches_offline_leace() -> None:
    values, labels, fitting_indices = synthetic_states()
    eraser = fit_leace(values, fitting_indices, labels[fitting_indices], class_count=4)
    intervention = intervention_from_leace(eraser)
    residual_states = torch.from_numpy(values[:6].reshape(2, 3, 12)).to(torch.float64)
    positions = torch.tensor([1, 2])

    edited_states = edit_selected_states(residual_states, positions, intervention)
    selected_values = residual_states[torch.arange(2), positions].numpy()
    expected_values = transform_with_leace(selected_values, eraser)

    np.testing.assert_allclose(
        edited_states[torch.arange(2), positions].numpy(),
        expected_values,
        atol=1e-10,
    )


def test_magnitude_matching_uses_only_supplied_fitting_values() -> None:
    fitting_values = np.arange(24, dtype=np.float64).reshape(6, 4)
    reference_values = fitting_values + 2.0
    control_values = fitting_values - 0.5

    scale = magnitude_matching_scale(
        fitting_values,
        reference_values,
        control_values,
    )

    assert scale == 4.0
    assert set(inspect.signature(magnitude_matching_scale).parameters) == {
        "fitting_values",
        "reference_fitting_values",
        "control_fitting_values",
    }


def test_magnitude_matched_random_control_matches_fitting_distortion() -> None:
    values, _, fitting_indices = synthetic_states()
    fitting_mean = values[fitting_indices].mean(axis=0)
    reference = intervention_from_random_projection(fitting_mean, 12, 3, random_seed=2)
    reference_values = transform_states(values, with_displacement_scale(reference, 2.5))
    control = intervention_from_random_projection(fitting_mean, 12, 3, random_seed=3)
    control_values = transform_states(values, control)
    scale = magnitude_matching_scale(
        values[fitting_indices],
        reference_values[fitting_indices],
        control_values[fitting_indices],
    )
    matched_control = with_displacement_scale(control, scale)
    matched_values = transform_states(values, matched_control)

    reference_change = mean_squared_change(
        values[fitting_indices], reference_values[fitting_indices]
    )
    matched_change = mean_squared_change(
        values[fitting_indices], matched_values[fitting_indices]
    )
    np.testing.assert_allclose(reference_change, matched_change, rtol=1e-12)


def test_every_random_displacement_has_rank_three() -> None:
    for random_seed in range(2, 12):
        intervention = intervention_from_random_projection(
            np.zeros(16),
            residual_width=16,
            displacement_rank=3,
            random_seed=random_seed,
        )
        assert displacement_rank(intervention) == 3


def test_model_execution_interfaces_cannot_receive_style_labels() -> None:
    for function in (edit_selected_states, run_intervention_logits):
        parameter_names = set(inspect.signature(function).parameters)
        assert "style_labels" not in parameter_names
        assert "test_labels" not in parameter_names


def test_paired_comparison_preserves_session_alignment() -> None:
    style_losses = np.array([5.0, 2.0, 8.0, 1.0])
    control_losses = np.array([4.0, 3.0, 6.0, 1.0])

    measurement = comparison_measurement(style_losses, control_losses, bootstrap_seed=7)

    assert measurement["mean_difference"] == 0.5


def test_measurement_finiteness_rejects_nonfinite_nested_values() -> None:
    assert all_measurement_numbers_finite({"values": [1.0, {"loss": 2.0}]})
    assert not all_measurement_numbers_finite({"values": [1.0, {"loss": np.nan}]})
    assert not all_measurement_numbers_finite({"loss": np.inf})


def test_real_checkpoint_intervention_changes_finite_downstream_logits() -> None:
    checkpoint_path = Path("training/checkpoints/seed0/step_008439.pt")
    token_path = Path("data/full/val_tokens.npy")
    vocabulary_path = Path("data/full/vocab.json")
    if not checkpoint_path.exists() or not token_path.exists() or not vocabulary_path.exists():
        raise AssertionError("Downloaded probing artifacts are required for the integration check")

    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))["vocab"]
    token_rows = np.load(token_path, mmap_mode="r")[:3]
    opponent_action_ids = tuple(vocabulary.index(action_name) for action_name in OPPONENT_ACTIONS)
    positions, targets = find_next_opponent_targets(
        token_rows,
        vocabulary.index("[HAND]"),
        opponent_action_ids,
    )
    model, checkpoint_state = load_checkpoint(str(checkpoint_path), device="cpu")
    states, untouched_logits = extract_predictor_states_and_logits(
        model,
        token_rows,
        positions,
        batch_size=3,
    )
    intervention = intervention_from_random_projection(
        states.mean(axis=0),
        residual_width=256,
        displacement_rank=3,
        random_seed=2,
    )
    changed_logits = run_intervention_logits(
        model,
        token_rows,
        positions,
        intervention,
        batch_size=3,
    )

    assert states.shape == (3, 256)
    assert untouched_logits.shape == changed_logits.shape == (3, 68)
    assert targets.shape == (3,)
    assert checkpoint_state["step"] == 8439
    assert np.isfinite(changed_logits).all()
    assert np.max(np.abs(changed_logits - untouched_logits)) > 0
