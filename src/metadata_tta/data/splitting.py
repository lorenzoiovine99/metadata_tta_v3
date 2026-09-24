from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import (
    train_test_split,
)


@dataclass(frozen=True)
class SplitIndices:
    """
    Deterministic train / validation / test indices for one year.
    """

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def _validate_fraction(
    value: float,
    *,
    name: str,
    allow_zero: bool,
) -> float:

    value = float(
        value
    )

    lower_ok = (
        value >= 0.0
        if allow_zero
        else value > 0.0
    )

    if (
        not lower_ok
        or value >= 1.0
    ):
        interval = (
            "[0, 1)"
            if allow_zero
            else "(0, 1)"
        )

        raise ValueError(
            f"{name} must be in {interval}, "
            f"got {value}."
        )

    return value


def _n_test_samples(
    n_samples: int,
    test_size: float,
) -> int:
    """
    Match sklearn's effective sample count for float test_size:
    ceil(test_size * n_samples).
    """

    return int(
        np.ceil(
            float(test_size)
            * int(n_samples)
        )
    )


def _safe_stratify_labels(
    labels: np.ndarray,
    *,
    test_size: float,
) -> np.ndarray | None:
    """
    Return labels only when stratified splitting is feasible.

    Requirements:
        - at least two represented classes;
        - every class has at least two samples;
        - both resulting subsets can contain at least one
          sample from every represented class.

    Otherwise return None and use deterministic shuffled
    splitting without stratification.
    """

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    n_samples = int(
        len(labels)
    )

    if n_samples < 2:
        return None

    _, counts = np.unique(
        labels,
        return_counts=True,
    )

    n_classes = int(
        len(counts)
    )

    if n_classes < 2:
        return None

    if int(
        counts.min()
    ) < 2:
        return None

    n_test = _n_test_samples(
        n_samples=n_samples,
        test_size=test_size,
    )

    n_train = (
        n_samples
        - n_test
    )

    if n_test < n_classes:
        return None

    if n_train < n_classes:
        return None

    return labels


def _split_once(
    indices: np.ndarray,
    labels: np.ndarray,
    *,
    test_size: float,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    stratify = (
        _safe_stratify_labels(
            labels,
            test_size=test_size,
        )
    )

    train_indices, test_indices = (
        train_test_split(
            indices,
            test_size=float(
                test_size
            ),
            random_state=int(
                seed
            ),
            shuffle=True,
            stratify=stratify,
        )
    )

    return (
        np.asarray(
            train_indices,
            dtype=np.int64,
        ),
        np.asarray(
            test_indices,
            dtype=np.int64,
        ),
    )


def build_split_indices(
    y_main: np.ndarray,
    test_size: float,
    validation_size_within_train: float,
    seed: int,
) -> SplitIndices:
    """
    Build one deterministic yearly 70 / 10 / 20-style split.

    With:

        test_size = 0.20
        validation_size_within_train = 0.125

    the effective fractions are:

        train       = 70%
        validation  = 10%
        test        = 20%

    split_seed controls membership.
    """

    y_main = np.asarray(
        y_main,
        dtype=np.int64,
    ).reshape(-1)

    n_samples = int(
        len(y_main)
    )

    if n_samples < 3:
        raise ValueError(
            "At least three samples are required "
            "for the V3 train/validation/test protocol."
        )

    test_size = _validate_fraction(
        test_size,
        name="test_size",
        allow_zero=False,
    )

    validation_fraction = (
        _validate_fraction(
            validation_size_within_train,
            name=(
                "validation_size_within_train"
            ),
            allow_zero=True,
        )
    )

    indices = np.arange(
        n_samples,
        dtype=np.int64,
    )

    (
        development_indices,
        test_indices,
    ) = _split_once(
        indices=indices,
        labels=y_main,
        test_size=test_size,
        seed=seed,
    )

    if validation_fraction == 0.0:

        train_indices = (
            development_indices
        )

        validation_indices = (
            np.empty(
                0,
                dtype=np.int64,
            )
        )

    else:

        development_labels = (
            y_main[
                development_indices
            ]
        )

        (
            train_indices,
            validation_indices,
        ) = _split_once(
            indices=development_indices,
            labels=development_labels,
            test_size=validation_fraction,
            seed=seed,
        )

    split = SplitIndices(
        train=np.asarray(
            train_indices,
            dtype=np.int64,
        ),
        validation=np.asarray(
            validation_indices,
            dtype=np.int64,
        ),
        test=np.asarray(
            test_indices,
            dtype=np.int64,
        ),
    )

    _verify_split(
        split=split,
        n_samples=n_samples,
    )

    return split


def _verify_split(
    *,
    split: SplitIndices,
    n_samples: int,
) -> None:

    train = set(
        map(
            int,
            split.train,
        )
    )

    validation = set(
        map(
            int,
            split.validation,
        )
    )

    test = set(
        map(
            int,
            split.test,
        )
    )

    if train & validation:
        raise RuntimeError(
            "Train and validation splits overlap."
        )

    if train & test:
        raise RuntimeError(
            "Train and test splits overlap."
        )

    if validation & test:
        raise RuntimeError(
            "Validation and test splits overlap."
        )

    expected = set(
        range(
            int(n_samples)
        )
    )

    observed = (
        train
        | validation
        | test
    )

    if observed != expected:
        raise RuntimeError(
            "Split indices do not reconstruct "
            "the complete dataset."
        )


def materialize_split(
    X: np.ndarray,
    y_main: np.ndarray,
    y_aux: np.ndarray,
    split: SplitIndices,
) -> dict[str, np.ndarray]:

    return {
        "X_train":
            X[split.train],

        "y_main_train":
            y_main[split.train],

        "y_aux_train":
            y_aux[split.train],

        "X_validation":
            X[split.validation],

        "y_main_validation":
            y_main[split.validation],

        "y_aux_validation":
            y_aux[split.validation],

        "X_test":
            X[split.test],

        "y_main_test":
            y_main[split.test],

        "y_aux_test":
            y_aux[split.test],
    }


def split_year_data(
    X: np.ndarray,
    y_main: np.ndarray,
    y_aux: np.ndarray,
    test_size: float,
    validation_size_within_train: float,
    seed: int,
) -> dict[str, np.ndarray]:

    split = build_split_indices(
        y_main=y_main,
        test_size=test_size,
        validation_size_within_train=(
            validation_size_within_train
        ),
        seed=seed,
    )

    return materialize_split(
        X=X,
        y_main=y_main,
        y_aux=y_aux,
        split=split,
    )


def concatenate_train_and_validation(
    split_data: dict[str, np.ndarray],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Legacy convenience utility.

    WARNING
    -------
    This returns train + validation, i.e. 80% under the
    standard 70/10/20 protocol.

    It must NOT be used for ordinary V3 source training,
    because V3 preserves validation for early stopping.
    """

    X = np.concatenate(
        [
            split_data["X_train"],
            split_data["X_validation"],
        ],
        axis=0,
    )

    y_main = np.concatenate(
        [
            split_data["y_main_train"],
            split_data[
                "y_main_validation"
            ],
        ],
        axis=0,
    )

    y_aux = np.concatenate(
        [
            split_data["y_aux_train"],
            split_data[
                "y_aux_validation"
            ],
        ],
        axis=0,
    )

    return (
        X,
        y_main,
        y_aux,
    )