"""Fit and measure opponent-style LEACE transformations."""

from dataclasses import dataclass

from concept_erasure import LeaceEraser
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


STYLE_CLASSIFIER_SEED = 0


@dataclass(frozen=True)
class OrthogonalEraser:
    center: np.ndarray
    directions: np.ndarray


def one_hot_concepts(fitting_labels: np.ndarray, class_count: int) -> torch.Tensor:
    label_tensor = torch.from_numpy(np.asarray(fitting_labels, dtype=np.int64))
    return torch.nn.functional.one_hot(label_tensor, class_count).to(torch.float64)


def fit_leace(
    feature_values: np.ndarray,
    fitting_indices: np.ndarray,
    fitting_labels: np.ndarray,
    class_count: int,
) -> LeaceEraser:
    fitting_values = torch.from_numpy(
        np.asarray(feature_values[fitting_indices], dtype=np.float64)
    )
    fitting_concepts = one_hot_concepts(fitting_labels, class_count)
    return LeaceEraser.fit(fitting_values, fitting_concepts)


def transform_with_leace(feature_values: np.ndarray, eraser: LeaceEraser) -> np.ndarray:
    value_tensor = torch.from_numpy(np.asarray(feature_values, dtype=np.float64))
    with torch.inference_mode():
        transformed_values = eraser(value_tensor)
    return transformed_values.cpu().numpy()


def random_rank_eraser(
    fitting_mean: np.ndarray,
    residual_width: int,
    erasure_rank: int,
    random_seed: int,
) -> OrthogonalEraser:
    random_generator = np.random.default_rng(random_seed)
    random_columns = random_generator.normal(size=(residual_width, erasure_rank))
    directions = np.linalg.qr(random_columns)[0].T
    return OrthogonalEraser(
        center=np.asarray(fitting_mean, dtype=np.float64),
        directions=np.asarray(directions, dtype=np.float64),
    )


def transformation_rank(eraser: LeaceEraser) -> int:
    identity = torch.eye(eraser.P.shape[0], dtype=eraser.P.dtype, device=eraser.P.device)
    return int(torch.linalg.matrix_rank(identity - eraser.P).item())


def maximum_class_mean_separation(
    feature_values: np.ndarray,
    fitting_indices: np.ndarray,
    fitting_labels: np.ndarray,
    class_count: int,
) -> float:
    fitting_values = feature_values[fitting_indices]
    class_means = np.stack(
        [fitting_values[fitting_labels == class_id].mean(axis=0) for class_id in range(class_count)]
    )
    pairwise_distances = np.linalg.norm(
        class_means[:, np.newaxis] - class_means[np.newaxis],
        axis=2,
    )
    return float(pairwise_distances.max())


def distortion_measurements(
    original_values: np.ndarray,
    transformed_values: np.ndarray,
    fitting_mean: np.ndarray,
    erasure_rank: int,
) -> dict[str, float | int]:
    changes = transformed_values - original_values
    original_norms = np.linalg.norm(original_values, axis=1)
    transformed_norms = np.linalg.norm(transformed_values, axis=1)
    cosine_denominators = original_norms * transformed_norms
    cosine_similarities = np.divide(
        np.sum(original_values * transformed_values, axis=1),
        cosine_denominators,
        out=np.ones_like(original_norms),
        where=cosine_denominators > 0,
    )
    centered_values = original_values - fitting_mean
    centered_energy = np.square(centered_values).sum()
    removed_energy = np.square(changes).sum()
    return {
        "mean_squared_activation_change": float(np.square(changes).mean()),
        "mean_cosine_similarity": float(cosine_similarities.mean()),
        "centered_activation_energy_fraction_removed": float(removed_energy / centered_energy),
        "transformation_rank": erasure_rank,
    }


def train_style_classifier(
    feature_values: np.ndarray,
    fitting_indices: np.ndarray,
    fitting_labels: np.ndarray,
) -> Pipeline:
    if len(fitting_indices) != len(fitting_labels):
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
                    random_state=STYLE_CLASSIFIER_SEED,
                ),
            ),
        ]
    )
    classifier.fit(feature_values[fitting_indices], fitting_labels)
    return classifier


def style_measurements(
    classifier: Pipeline,
    feature_values: np.ndarray,
    test_indices: np.ndarray,
    test_labels: np.ndarray,
    class_ids: np.ndarray,
) -> dict[str, object]:
    predictions = classifier.predict(feature_values[test_indices])
    return {
        "balanced_accuracy": float(balanced_accuracy_score(test_labels, predictions)),
        "macro_f1": float(f1_score(test_labels, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(
            test_labels,
            predictions,
            labels=class_ids,
        ).tolist(),
    }


def fresh_probe_measurements(
    feature_values: np.ndarray,
    style_labels: np.ndarray,
    fitting_indices: np.ndarray,
    test_indices: np.ndarray,
    class_ids: np.ndarray,
) -> dict[str, object]:
    classifier = train_style_classifier(
        feature_values,
        fitting_indices,
        style_labels[fitting_indices],
    )
    return style_measurements(
        classifier,
        feature_values,
        test_indices,
        style_labels[test_indices],
        class_ids,
    )
