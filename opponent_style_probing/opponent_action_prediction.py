"""Select opponent-action targets and measure next-token predictions."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as torch_functional

from opponent_style_probing.hand20_style_probe import find_hand21_positions


BLOCK4_RESIDUAL_HOOK = "blocks.4.hook_resid_post"
OPPONENT_ACTIONS = ("O:fold", "O:check", "O:call", "O:raise")


@dataclass(frozen=True)
class ActionPredictions:
    losses: np.ndarray
    correct_probabilities: np.ndarray
    restricted_correct: np.ndarray


def find_next_opponent_targets(
    token_rows: np.ndarray,
    hand_token_id: int,
    opponent_action_ids: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Select the first opponent action after twenty complete hands."""

    hand21_positions = find_hand21_positions(token_rows, hand_token_id)
    predictor_positions = np.empty(len(token_rows), dtype=np.int32)
    target_ids = np.empty(len(token_rows), dtype=np.int64)
    for session_index, hand21_position in enumerate(hand21_positions):
        suffix = token_rows[session_index, int(hand21_position) :]
        action_offsets = np.flatnonzero(np.isin(suffix, opponent_action_ids))
        if not len(action_offsets):
            raise ValueError(f"Session {session_index} has no opponent action after hand 20")
        target_position = int(hand21_position + action_offsets[0])
        predictor_positions[session_index] = target_position - 1
        target_ids[session_index] = token_rows[session_index, target_position]
    return predictor_positions, target_ids


def extract_predictor_states_and_logits(
    model,
    token_rows: np.ndarray,
    predictor_positions: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract block-4 states and untouched logits without receiving style labels."""

    residual_width = int(model.cfg.d_model)
    vocabulary_size = int(model.cfg.d_vocab)
    predictor_states = np.empty((len(token_rows), residual_width), dtype=np.float32)
    predictor_logits = np.empty((len(token_rows), vocabulary_size), dtype=np.float32)
    model_device = next(model.parameters()).device

    with torch.inference_mode():
        for batch_start in range(0, len(token_rows), batch_size):
            batch_stop = min(batch_start + batch_size, len(token_rows))
            batch_positions = predictor_positions[batch_start:batch_stop]
            batch_length = int(batch_positions.max()) + 1
            batch_tokens = torch.from_numpy(
                np.asarray(token_rows[batch_start:batch_stop, :batch_length], dtype=np.int64)
            ).to(model_device)
            logits, activation_cache = model.run_with_cache(
                batch_tokens,
                names_filter=lambda hook_name: hook_name == BLOCK4_RESIDUAL_HOOK,
                return_type="logits",
            )
            batch_rows = torch.arange(batch_stop - batch_start, device=model_device)
            position_indices = torch.from_numpy(batch_positions.astype(np.int64)).to(model_device)
            predictor_states[batch_start:batch_stop] = (
                activation_cache[BLOCK4_RESIDUAL_HOOK][batch_rows, position_indices]
                .float()
                .cpu()
                .numpy()
            )
            predictor_logits[batch_start:batch_stop] = (
                logits[batch_rows, position_indices].float().cpu().numpy()
            )

    if not np.isfinite(predictor_states).all() or not np.isfinite(predictor_logits).all():
        raise ValueError("Extracted states and logits must be finite")
    return predictor_states, predictor_logits


def action_predictions(
    predictor_logits: np.ndarray,
    target_ids: np.ndarray,
    opponent_action_ids: tuple[int, ...],
) -> ActionPredictions:
    logits_tensor = torch.from_numpy(predictor_logits).float()
    targets_tensor = torch.from_numpy(target_ids).long()
    losses = torch_functional.cross_entropy(logits_tensor, targets_tensor, reduction="none").numpy()
    probabilities = logits_tensor.softmax(dim=1)
    row_indices = torch.arange(len(target_ids))
    correct_probabilities = probabilities[row_indices, targets_tensor].numpy()
    opponent_logits = predictor_logits[:, opponent_action_ids]
    restricted_ids = np.asarray(opponent_action_ids)[opponent_logits.argmax(axis=1)]
    return ActionPredictions(
        losses=losses,
        correct_probabilities=correct_probabilities,
        restricted_correct=restricted_ids == target_ids,
    )


def prediction_measurements(predictions: ActionPredictions) -> dict[str, float]:
    return {
        "full_vocabulary_cross_entropy": float(predictions.losses.mean()),
        "correct_action_probability": float(predictions.correct_probabilities.mean()),
        "restricted_opponent_action_accuracy": float(predictions.restricted_correct.mean()),
    }


def bootstrap_mean_difference(
    first_values: np.ndarray,
    second_values: np.ndarray,
    repetitions: int,
    random_seed: int,
) -> dict[str, float | int]:
    if first_values.shape != second_values.shape:
        raise ValueError("Paired measurements must have matching shapes")
    paired_differences = first_values - second_values
    random_generator = np.random.default_rng(random_seed)
    sampled_means = np.empty(repetitions, dtype=np.float64)
    for repetition_index in range(repetitions):
        sampled_indices = random_generator.integers(
            0,
            len(paired_differences),
            len(paired_differences),
        )
        sampled_means[repetition_index] = paired_differences[sampled_indices].mean()
    lower_bound, upper_bound = np.quantile(sampled_means, [0.025, 0.975])
    return {
        "mean_difference": float(paired_differences.mean()),
        "confidence_level": 0.95,
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "repetitions": repetitions,
    }
