import inspect
from pathlib import Path
from unittest.mock import patch

import numpy as np

from opponent_style_probing.hand20_style_probe import (
    extract_hand21_residual_states,
    find_hand21_positions,
    fit_probe_comparisons,
    fit_style_classifier,
    make_session_partition,
    opponent_action_features,
)


HAND_TOKEN_ID = 1
HERO_CHECK_ID = 9
OPPONENT_ACTION_IDS = (12, 13, 14, 15)
OUTCOME_ID = 6


def session_tokens(opponent_actions: list[int]) -> np.ndarray:
    if len(opponent_actions) != 50:
        raise ValueError("A test session requires exactly 50 opponent actions")
    tokens = np.zeros(1024, dtype=np.uint16)
    write_position = 0
    for opponent_action_id in opponent_actions:
        hand_tokens = [HAND_TOKEN_ID, HERO_CHECK_ID, opponent_action_id, OUTCOME_ID]
        tokens[write_position : write_position + len(hand_tokens)] = hand_tokens
        write_position += len(hand_tokens)
    return tokens


def test_selects_the_marker_after_twenty_complete_hands() -> None:
    token_rows = np.stack(
        [
            session_tokens([12] * 50),
            session_tokens([13] * 50),
        ]
    )
    positions = find_hand21_positions(token_rows, HAND_TOKEN_ID)
    np.testing.assert_array_equal(positions, [80, 80])
    assert np.all(token_rows[np.arange(2), positions] == HAND_TOKEN_ID)


def test_rejects_sessions_without_exactly_fifty_hands() -> None:
    malformed_session = session_tokens([12] * 50)
    malformed_session[196] = 0
    with np.testing.assert_raises_regex(ValueError, "49.*expected 50"):
        find_hand21_positions(malformed_session[np.newaxis, :], HAND_TOKEN_ID)


def test_action_counts_stop_before_hand_twenty_one() -> None:
    first_twenty_actions = list(OPPONENT_ACTION_IDS) * 5
    later_actions = [15] * 30
    token_rows = session_tokens(first_twenty_actions + later_actions)[np.newaxis, :]
    positions = find_hand21_positions(token_rows, HAND_TOKEN_ID)
    features = opponent_action_features(token_rows, positions, OPPONENT_ACTION_IDS)

    np.testing.assert_array_equal(features[0, :4], [5, 5, 5, 5])
    np.testing.assert_allclose(features[0, 4:], [0.25, 0.25, 0.25, 0.25])


def test_stratified_partition_is_disjoint_complete_and_deterministic() -> None:
    style_labels = np.tile(np.arange(4, dtype=np.int64), 25)
    first_partition = make_session_partition(style_labels, split_seed=7)
    second_partition = make_session_partition(style_labels, split_seed=7)

    assert len(first_partition.fitting_indices) == 70
    assert len(first_partition.test_indices) == 30
    assert np.intersect1d(
        first_partition.fitting_indices,
        first_partition.test_indices,
    ).size == 0
    np.testing.assert_array_equal(
        np.sort(np.concatenate((first_partition.fitting_indices, first_partition.test_indices))),
        np.arange(100),
    )
    assert set(style_labels[first_partition.fitting_indices]) == {0, 1, 2, 3}
    assert set(style_labels[first_partition.test_indices]) == {0, 1, 2, 3}
    np.testing.assert_array_equal(
        first_partition.fitting_indices,
        second_partition.fitting_indices,
    )
    np.testing.assert_array_equal(
        first_partition.test_indices,
        second_partition.test_indices,
    )


def test_linear_classifier_recovers_a_known_style_signal() -> None:
    random_generator = np.random.default_rng(5)
    style_labels = np.tile(np.arange(4, dtype=np.int64), 100)
    feature_values = random_generator.normal(0, 0.1, (400, 8)).astype(np.float32)
    feature_values[np.arange(400), style_labels] += 5.0
    partition = make_session_partition(style_labels, split_seed=3)

    result = fit_style_classifier(
        feature_values,
        style_labels,
        partition,
        np.arange(4),
        style_labels[partition.fitting_indices],
    )

    assert result.balanced_accuracy > 0.99
    assert result.macro_f1 > 0.99


def test_all_comparisons_receive_the_same_session_partition() -> None:
    random_generator = np.random.default_rng(8)
    style_labels = np.tile(np.arange(4, dtype=np.int64), 25)
    trained_states = random_generator.normal(size=(100, 16)).astype(np.float32)
    untrained_states = random_generator.normal(size=(100, 16)).astype(np.float32)
    count_features = random_generator.normal(size=(100, 8)).astype(np.float32)
    partition = make_session_partition(style_labels, split_seed=9)

    with patch(
        "opponent_style_probing.hand20_style_probe.fit_style_classifier",
        wraps=fit_style_classifier,
    ) as classifier_spy:
        fit_probe_comparisons(
            trained_states,
            untrained_states,
            count_features,
            style_labels,
            partition,
            np.arange(4),
            shuffle_repeats=2,
            shuffle_seed=10,
        )

    observed_partitions = [call.args[2] for call in classifier_spy.call_args_list]
    assert len(observed_partitions) == 5
    assert all(observed_partition is partition for observed_partition in observed_partitions)


def test_extraction_interface_cannot_receive_style_labels() -> None:
    parameter_names = set(inspect.signature(extract_hand21_residual_states).parameters)
    assert "style_labels" not in parameter_names


def test_both_checkpoints_yield_one_finite_width_256_state_per_session() -> None:
    checkpoint_cases = [
        (Path("training/checkpoints/seed0/step_008439.pt"), 8439),
        (Path("training/checkpoints/seed0/step_000000.pt"), 0),
    ]
    token_path = Path("data/full/val_tokens.npy")
    if not token_path.exists() or any(
        not checkpoint_path.exists() for checkpoint_path, expected_step in checkpoint_cases
    ):
        raise AssertionError("Downloaded probing artifacts are required for the integration check")
    token_rows = np.load(token_path, mmap_mode="r")[:3]
    positions = find_hand21_positions(token_rows, HAND_TOKEN_ID)

    for checkpoint_path, expected_step in checkpoint_cases:
        residual_states, identity = extract_hand21_residual_states(
            checkpoint_path,
            token_rows,
            positions,
            batch_size=3,
            device="cpu",
        )

        assert residual_states.shape == (3, 256)
        assert np.isfinite(residual_states).all()
        assert identity["step"] == expected_step
