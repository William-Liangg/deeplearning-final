"""Test whether opponent style is linearly readable after twenty hands."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from training.train import load_checkpoint


COMPLETED_HANDS = 20
EXPECTED_HANDS = 50
TRAINING_FRACTION = 0.70
SPLIT_SEED = 0
SHUFFLE_REPEATS = 20
BOOTSTRAP_REPEATS = 1000
FINAL_RESIDUAL_HOOK = "blocks.5.hook_resid_post"
OPPONENT_ACTIONS = ("O:fold", "O:check", "O:call", "O:raise")
TRAINED_CHECKPOINT_PATH = Path("training/checkpoints/seed0/step_008439.pt")
UNTRAINED_CHECKPOINT_PATH = Path("training/checkpoints/seed0/step_000000.pt")
TOKEN_PATH = Path("data/full/val_tokens.npy")
STYLE_PATH = Path("data/full/val_styles.npy")
VOCABULARY_PATH = Path("data/full/vocab.json")
MEASUREMENT_PATH = Path("opponent_style_probing/measurements/hand20_seed0.json")


@dataclass(frozen=True)
class SessionPartition:
    fitting_indices: np.ndarray
    test_indices: np.ndarray


@dataclass(frozen=True)
class ClassifierResult:
    balanced_accuracy: float
    macro_f1: float
    confusion: list[list[int]]
    predictions: np.ndarray


@dataclass(frozen=True)
class ProbeComparisons:
    trained: ClassifierResult
    untrained: ClassifierResult
    action_counts: ClassifierResult
    shuffled_scores: list[float]


def read_vocabulary(vocabulary_path: Path) -> tuple[list[str], list[str]]:
    vocabulary_document = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    return vocabulary_document["vocab"], vocabulary_document["styles"]


def find_hand21_positions(token_rows: np.ndarray, hand_token_id: int) -> np.ndarray:
    """Return the token position of hand 21 and reject malformed sessions."""

    selected_positions = np.empty(len(token_rows), dtype=np.int32)
    for session_index, token_row in enumerate(token_rows):
        hand_positions = np.flatnonzero(token_row == hand_token_id)
        if len(hand_positions) != EXPECTED_HANDS:
            raise ValueError(
                f"Session {session_index} has {len(hand_positions)} [HAND] markers; "
                f"expected {EXPECTED_HANDS}"
            )
        selected_positions[session_index] = hand_positions[COMPLETED_HANDS]
    return selected_positions


def opponent_action_features(
    token_rows: np.ndarray,
    hand21_positions: np.ndarray,
    opponent_action_ids: tuple[int, ...],
) -> np.ndarray:
    """Count opponent actions strictly before hand 21, then append proportions."""

    action_counts = np.zeros((len(token_rows), len(opponent_action_ids)), dtype=np.float32)
    for session_index, hand21_position in enumerate(hand21_positions):
        observed_prefix = token_rows[session_index, : int(hand21_position)]
        action_counts[session_index] = [
            np.count_nonzero(observed_prefix == action_id)
            for action_id in opponent_action_ids
        ]
    totals = action_counts.sum(axis=1, keepdims=True)
    proportions = np.divide(
        action_counts,
        totals,
        out=np.zeros_like(action_counts),
        where=totals > 0,
    )
    return np.concatenate((action_counts, proportions), axis=1)


def make_session_partition(style_labels: np.ndarray, split_seed: int) -> SessionPartition:
    session_indices = np.arange(len(style_labels), dtype=np.int64)
    fitting_indices, test_indices = train_test_split(
        session_indices,
        train_size=TRAINING_FRACTION,
        random_state=split_seed,
        stratify=style_labels,
    )
    if np.intersect1d(fitting_indices, test_indices).size:
        raise AssertionError("Classifier-fitting and test sessions overlap")
    expected_styles = set(np.unique(style_labels).tolist())
    if set(np.unique(style_labels[fitting_indices]).tolist()) != expected_styles:
        raise AssertionError("Classifier-fitting sessions do not contain every style")
    if set(np.unique(style_labels[test_indices]).tolist()) != expected_styles:
        raise AssertionError("Test sessions do not contain every style")
    return SessionPartition(
        fitting_indices=np.sort(fitting_indices),
        test_indices=np.sort(test_indices),
    )


def checkpoint_identity(checkpoint_path: Path, checkpoint_state: dict) -> dict[str, object]:
    file_digest = sha256()
    with checkpoint_path.open("rb") as checkpoint_file:
        for file_chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            file_digest.update(file_chunk)
    return {
        "path": str(checkpoint_path),
        "sha256": file_digest.hexdigest(),
        "step": int(checkpoint_state["step"]),
        "epoch": float(checkpoint_state["epoch"]),
    }


def extract_hand21_residual_states(
    checkpoint_path: Path,
    token_rows: np.ndarray,
    hand21_positions: np.ndarray,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Extract one last-layer state per session without receiving style labels."""

    model, checkpoint_state = load_checkpoint(str(checkpoint_path), device=device)
    residual_width = int(model.cfg.d_model)
    if FINAL_RESIDUAL_HOOK not in model.hook_dict:
        raise KeyError(f"Checkpoint does not expose {FINAL_RESIDUAL_HOOK}")
    residual_states = np.empty((len(token_rows), residual_width), dtype=np.float32)
    model_device = next(model.parameters()).device

    with torch.inference_mode():
        for batch_start in range(0, len(token_rows), batch_size):
            batch_stop = min(batch_start + batch_size, len(token_rows))
            batch_positions = hand21_positions[batch_start:batch_stop]
            batch_length = int(batch_positions.max()) + 1
            batch_tokens = torch.from_numpy(
                np.asarray(
                    token_rows[batch_start:batch_stop, :batch_length],
                    dtype=np.int64,
                )
            ).to(model_device)
            _, activation_cache = model.run_with_cache(
                batch_tokens,
                names_filter=lambda hook_name: hook_name == FINAL_RESIDUAL_HOOK,
                return_type=None,
            )
            batch_row_indices = torch.arange(batch_stop - batch_start, device=model_device)
            position_indices = torch.from_numpy(batch_positions.astype(np.int64)).to(model_device)
            selected_states = activation_cache[FINAL_RESIDUAL_HOOK][
                batch_row_indices,
                position_indices,
            ]
            residual_states[batch_start:batch_stop] = selected_states.float().cpu().numpy()

    if not np.isfinite(residual_states).all():
        raise ValueError("Residual states contain non-finite values")
    return residual_states, checkpoint_identity(checkpoint_path, checkpoint_state)


def fit_style_classifier(
    feature_values: np.ndarray,
    style_labels: np.ndarray,
    partition: SessionPartition,
    class_ids: np.ndarray,
    fitting_labels: np.ndarray,
) -> ClassifierResult:
    if len(fitting_labels) != len(partition.fitting_indices):
        raise ValueError("Fitting labels must align with fitting sessions")
    classifier = Pipeline(
        steps=[
            ("standardizer", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    l1_ratio=0.0,
                    C=1.0,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=SPLIT_SEED,
                ),
            ),
        ]
    )
    classifier.fit(feature_values[partition.fitting_indices], fitting_labels)
    predictions = classifier.predict(feature_values[partition.test_indices])
    test_labels = style_labels[partition.test_indices]
    return ClassifierResult(
        balanced_accuracy=float(balanced_accuracy_score(test_labels, predictions)),
        macro_f1=float(f1_score(test_labels, predictions, average="macro")),
        confusion=confusion_matrix(test_labels, predictions, labels=class_ids).tolist(),
        predictions=predictions,
    )


def shuffled_label_distribution(
    trained_states: np.ndarray,
    style_labels: np.ndarray,
    partition: SessionPartition,
    class_ids: np.ndarray,
    repetitions: int,
    random_seed: int,
) -> list[float]:
    random_generator = np.random.default_rng(random_seed)
    fitting_labels = style_labels[partition.fitting_indices]
    null_scores: list[float] = []
    for repetition_index in range(repetitions):
        shuffled_labels = random_generator.permutation(fitting_labels)
        result = fit_style_classifier(
            trained_states,
            style_labels,
            partition,
            class_ids,
            fitting_labels=shuffled_labels,
        )
        null_scores.append(result.balanced_accuracy)
    return null_scores


def fit_probe_comparisons(
    trained_states: np.ndarray,
    untrained_states: np.ndarray,
    count_features: np.ndarray,
    style_labels: np.ndarray,
    partition: SessionPartition,
    class_ids: np.ndarray,
    shuffle_repeats: int,
    shuffle_seed: int,
) -> ProbeComparisons:
    """Fit every comparison through one explicit, shared session partition."""

    fitting_style_labels = style_labels[partition.fitting_indices]
    trained_result = fit_style_classifier(
        trained_states, style_labels, partition, class_ids, fitting_style_labels
    )
    untrained_result = fit_style_classifier(
        untrained_states, style_labels, partition, class_ids, fitting_style_labels
    )
    count_result = fit_style_classifier(
        count_features, style_labels, partition, class_ids, fitting_style_labels
    )
    shuffled_scores = shuffled_label_distribution(
        trained_states,
        style_labels,
        partition,
        class_ids,
        shuffle_repeats,
        shuffle_seed,
    )
    return ProbeComparisons(
        trained=trained_result,
        untrained=untrained_result,
        action_counts=count_result,
        shuffled_scores=shuffled_scores,
    )


def bootstrap_accuracy_difference(
    test_labels: np.ndarray,
    trained_predictions: np.ndarray,
    count_predictions: np.ndarray,
    repetitions: int,
    random_seed: int,
) -> dict[str, object]:
    random_generator = np.random.default_rng(random_seed)
    sample_count = len(test_labels)
    observed_difference = float(
        balanced_accuracy_score(test_labels, trained_predictions)
        - balanced_accuracy_score(test_labels, count_predictions)
    )
    sampled_differences = np.empty(repetitions, dtype=np.float64)
    for repetition_index in range(repetitions):
        sampled_indices = random_generator.integers(0, sample_count, size=sample_count)
        sampled_labels = test_labels[sampled_indices]
        sampled_differences[repetition_index] = (
            balanced_accuracy_score(sampled_labels, trained_predictions[sampled_indices])
            - balanced_accuracy_score(sampled_labels, count_predictions[sampled_indices])
        )
    lower_bound, upper_bound = np.quantile(sampled_differences, [0.025, 0.975])
    return {
        "observed_difference": observed_difference,
        "confidence_level": 0.95,
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "repetitions": repetitions,
    }


def result_measurements(result: ClassifierResult) -> dict[str, object]:
    return {
        "balanced_accuracy": result.balanced_accuracy,
        "macro_f1": result.macro_f1,
        "confusion_matrix": result.confusion,
    }


def run_hand20_style_probe(batch_size: int, device: str) -> dict[str, object]:
    vocabulary, style_names = read_vocabulary(VOCABULARY_PATH)
    token_rows = np.load(TOKEN_PATH, mmap_mode="r")
    style_labels = np.load(STYLE_PATH)
    if token_rows.ndim != 2 or token_rows.shape[1] != 1024:
        raise ValueError("Validation tokens must have shape (sessions, 1024)")
    if style_labels.shape != (len(token_rows),):
        raise ValueError("Style labels must contain one value per session")
    class_ids = np.arange(len(style_names), dtype=np.int64)
    if set(np.unique(style_labels).tolist()) != set(class_ids.tolist()):
        raise ValueError("Style labels do not match vocab.json style indices")

    hand_token_id = vocabulary.index("[HAND]")
    hand21_positions = find_hand21_positions(token_rows, hand_token_id)
    opponent_action_ids = tuple(vocabulary.index(action_name) for action_name in OPPONENT_ACTIONS)
    count_features = opponent_action_features(
        token_rows,
        hand21_positions,
        opponent_action_ids,
    )
    partition = make_session_partition(style_labels, SPLIT_SEED)

    trained_states, trained_identity = extract_hand21_residual_states(
        TRAINED_CHECKPOINT_PATH,
        token_rows,
        hand21_positions,
        batch_size,
        device,
    )
    untrained_states, untrained_identity = extract_hand21_residual_states(
        UNTRAINED_CHECKPOINT_PATH,
        token_rows,
        hand21_positions,
        batch_size,
        device,
    )
    if trained_states.shape[1] != 256 or untrained_states.shape[1] != 256:
        raise ValueError("The approved experiment requires residual width 256")

    comparisons = fit_probe_comparisons(
        trained_states,
        untrained_states,
        count_features,
        style_labels,
        partition,
        class_ids,
        SHUFFLE_REPEATS,
        SPLIT_SEED + 1,
    )
    test_labels = style_labels[partition.test_indices]
    bootstrap_interval = bootstrap_accuracy_difference(
        test_labels,
        comparisons.trained.predictions,
        comparisons.action_counts.predictions,
        BOOTSTRAP_REPEATS,
        SPLIT_SEED + 2,
    )

    measurements: dict[str, object] = {
        "question": "After observing 20 hands, can opponent style be identified from the trained model's internal state?",
        "inputs": {
            "trained_checkpoint": trained_identity,
            "untrained_checkpoint": untrained_identity,
            "tokens": str(TOKEN_PATH),
            "styles": str(STYLE_PATH),
            "vocabulary": str(VOCABULARY_PATH),
        },
        "experiment": {
            "sessions": len(token_rows),
            "classifier_fitting_sessions": len(partition.fitting_indices),
            "test_sessions": len(partition.test_indices),
            "completed_hands": COMPLETED_HANDS,
            "activation_hook": FINAL_RESIDUAL_HOOK,
            "residual_width": trained_states.shape[1],
            "split_seed": SPLIT_SEED,
            "batch_size": batch_size,
            "device": device,
            "shuffle_repeats": SHUFFLE_REPEATS,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "style_names": style_names,
            "chance_balanced_accuracy": 1.0 / len(style_names),
            "classifier": "standardization followed by L2 multinomial logistic regression",
            "action_count_features": [
                "fold_count",
                "check_count",
                "call_count",
                "raise_count",
                "fold_proportion",
                "check_proportion",
                "call_proportion",
                "raise_proportion",
            ],
        },
        "measurements": {
            "trained_activations": result_measurements(comparisons.trained),
            "untrained_activations": result_measurements(comparisons.untrained),
            "action_counts": result_measurements(comparisons.action_counts),
            "shuffled_labels": {
                "repetitions": SHUFFLE_REPEATS,
                "balanced_accuracies": comparisons.shuffled_scores,
                "mean": float(np.mean(comparisons.shuffled_scores)),
                "lower_2_5_percentile": float(np.quantile(comparisons.shuffled_scores, 0.025)),
                "upper_97_5_percentile": float(np.quantile(comparisons.shuffled_scores, 0.975)),
            },
            "trained_minus_action_counts": bootstrap_interval,
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformer_lens": version("transformer-lens"),
            "numpy": np.__version__,
            "scikit_learn": version("scikit-learn"),
        },
    }
    MEASUREMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEASUREMENT_PATH.write_text(json.dumps(measurements, indent=2), encoding="utf-8")
    return measurements


def main() -> None:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--batch-size", type=int, default=64)
    argument_parser.add_argument("--device", default="cpu")
    arguments = argument_parser.parse_args()
    measurements = run_hand20_style_probe(arguments.batch_size, arguments.device)
    print(json.dumps(measurements["measurements"], indent=2))


if __name__ == "__main__":
    main()
