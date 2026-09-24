from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np

from metadata_tta.data import (
    DataBundle,
    SplitIndices,
    build_split_indices,
)


class SplitName(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ProtocolPhase(str, Enum):
    PSEUDO_SOURCE = "pseudo_source"
    PSEUDO_OOD = "pseudo_ood"
    FINAL_SOURCE = "final_source"
    REAL_OOD = "real_ood"


@dataclass(frozen=True)
class YearSplit:
    """
    Complete deterministic split information for one year.

    The original full-year arrays remain inside DataBundle.
    This object stores only the indices defining the
    70 / 10 / 20 protocol split.
    """

    year: int
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def indices(
        self,
        split_name: SplitName | str,
    ) -> np.ndarray:

        split_name = SplitName(
            split_name
        )

        if split_name is SplitName.TRAIN:
            return self.train

        if split_name is SplitName.VALIDATION:
            return self.validation

        if split_name is SplitName.TEST:
            return self.test

        raise RuntimeError(
            f"Unhandled split: {split_name}"
        )


@dataclass(frozen=True)
class StreamSlice:
    """
    Materialized chronological stream for one year.

    y_main is present because the evaluator needs it to score
    predictions after they have been produced.

    TTA methods themselves must never receive y_main.
    """

    year: int
    split_name: SplitName
    X: np.ndarray
    y_main: np.ndarray
    y_aux: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(
            len(self.X)
        )

    def as_evaluator_tuple(
        self,
    ) -> tuple[
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        return (
            self.year,
            self.X,
            self.y_main,
            self.y_aux,
        )


@dataclass(frozen=True)
class SupervisedSlice:
    """
    Train/validation arrays for supervised learning in one year.
    """

    year: int

    X_train: np.ndarray
    y_main_train: np.ndarray
    y_aux_train: np.ndarray

    X_validation: np.ndarray
    y_main_validation: np.ndarray
    y_aux_validation: np.ndarray


@dataclass(frozen=True)
class ProtocolYears:
    pseudo_source: tuple[int, ...]
    pseudo_ood: tuple[int, ...]
    final_source: tuple[int, ...]
    real_ood: tuple[int, ...]


def _inclusive_years(
    start_year: int,
    end_year: int,
) -> tuple[int, ...]:

    start_year = int(
        start_year
    )

    end_year = int(
        end_year
    )

    if start_year > end_year:
        raise ValueError(
            f"Invalid year range: "
            f"{start_year} > {end_year}."
        )

    return tuple(
        range(
            start_year,
            end_year + 1,
        )
    )


def _read_phase_years(
    protocol_config: dict,
    phase: ProtocolPhase,
) -> tuple[int, ...]:

    section = protocol_config.get(
        phase.value
    )

    if not isinstance(
        section,
        dict,
    ):
        raise ValueError(
            f"protocol.{phase.value} "
            "must be a mapping."
        )

    if (
        "start_year" not in section
        or "end_year" not in section
    ):
        raise ValueError(
            f"protocol.{phase.value} requires "
            "start_year and end_year."
        )

    return _inclusive_years(
        start_year=int(
            section["start_year"]
        ),
        end_year=int(
            section["end_year"]
        ),
    )


def build_protocol_years(
    protocol_config: dict,
) -> ProtocolYears:
    """
    Parse and validate the four temporal ranges.

    Required semantics:

        pseudo_source
            years used for model/aux tuning.

        pseudo_ood
            validation streams used for TTA tuning.

        final_source
            source years used for final retraining.

        real_ood
            held-out OOD years used for final evaluation.
    """

    pseudo_source = _read_phase_years(
        protocol_config,
        ProtocolPhase.PSEUDO_SOURCE,
    )

    pseudo_ood = _read_phase_years(
        protocol_config,
        ProtocolPhase.PSEUDO_OOD,
    )

    final_source = _read_phase_years(
        protocol_config,
        ProtocolPhase.FINAL_SOURCE,
    )

    real_ood = _read_phase_years(
        protocol_config,
        ProtocolPhase.REAL_OOD,
    )

    _assert_disjoint(
        pseudo_source,
        pseudo_ood,
        left_name="pseudo_source",
        right_name="pseudo_ood",
    )

    _assert_disjoint(
        final_source,
        real_ood,
        left_name="final_source",
        right_name="real_ood",
    )

    if max(
        pseudo_source
    ) >= min(
        pseudo_ood
    ):
        raise ValueError(
            "pseudo_source must end before "
            "pseudo_ood begins."
        )

    if max(
        final_source
    ) >= min(
        real_ood
    ):
        raise ValueError(
            "final_source must end before "
            "real_ood begins."
        )

    expected_final_source = tuple(
        sorted(
            set(pseudo_source)
            | set(pseudo_ood)
        )
    )

    if final_source != expected_final_source:
        raise ValueError(
            "V3 protocol requires final_source to equal "
            "pseudo_source + pseudo_ood. "
            f"Expected {expected_final_source}, "
            f"got {final_source}."
        )

    return ProtocolYears(
        pseudo_source=pseudo_source,
        pseudo_ood=pseudo_ood,
        final_source=final_source,
        real_ood=real_ood,
    )


def _assert_disjoint(
    left: Iterable[int],
    right: Iterable[int],
    *,
    left_name: str,
    right_name: str,
) -> None:

    overlap = (
        set(left)
        & set(right)
    )

    if overlap:
        raise RuntimeError(
            f"Temporal leakage: {left_name} and "
            f"{right_name} overlap in years "
            f"{sorted(overlap)}."
        )


class V3Protocol:
    """
    Single source of truth for temporal ranges and yearly splits.

    Important
    ---------
    experiment_seed:
        controls model initialization, DataLoader shuffling,
        dropout, and other experiment randomness.

    split_seed:
        controls ONLY train / validation / test membership.

    This class only receives split_seed.
    """

    def __init__(
        self,
        *,
        bundle: DataBundle,
        dataset_config: dict,
        protocol_config: dict,
        split_seed: int,
    ) -> None:

        self.bundle = bundle

        self.dataset_config = dict(
            dataset_config
        )

        self.protocol_config = dict(
            protocol_config
        )

        self.split_seed = int(
            split_seed
        )

        self.years = build_protocol_years(
            self.protocol_config
        )

        split_config = (
            self.dataset_config.get(
                "split"
            )
        )

        if not isinstance(
            split_config,
            dict,
        ):
            raise ValueError(
                "dataset.split must be a mapping."
            )

        self.test_size = float(
            split_config[
                "test_size"
            ]
        )

        self.validation_size_within_train = float(
            split_config[
                "validation_size_within_train"
            ]
        )

        self._validate_split_fractions()

        self._validate_required_years()

        self._splits = (
            self._build_all_required_splits()
        )

        self.verify_split_disjointness()

    def _validate_split_fractions(
        self,
    ) -> None:

        if not (
            0.0
            < self.test_size
            < 1.0
        ):
            raise ValueError(
                "dataset.split.test_size must "
                "be in (0, 1)."
            )

        if not (
            0.0
            <= self.validation_size_within_train
            < 1.0
        ):
            raise ValueError(
                "dataset.split."
                "validation_size_within_train "
                "must be in [0, 1)."
            )

        if (
            self.validation_size_within_train
            <= 0.0
        ):
            raise ValueError(
                "V3 requires a non-empty validation split."
            )

    def _validate_required_years(
        self,
    ) -> None:

        required = set(
            self.years.final_source
        ) | set(
            self.years.real_ood
        )

        available = set(
            self.bundle.years
        )

        missing = sorted(
            required - available
        )

        if missing:
            raise ValueError(
                "Dataset is missing protocol years: "
                f"{missing}"
            )

    def _build_split(
        self,
        year: int,
    ) -> YearSplit:

        data = self.bundle.year(
            year
        )

        split: SplitIndices = (
            build_split_indices(
                y_main=data.y_main,
                test_size=self.test_size,
                validation_size_within_train=(
                    self.validation_size_within_train
                ),
                seed=self.split_seed,
            )
        )

        return YearSplit(
            year=int(year),
            train=np.asarray(
                split.train,
                dtype=np.int64,
            ),
            validation=np.asarray(
                split.validation,
                dtype=np.int64,
            ),
            test=np.asarray(
                split.test,
                dtype=np.int64,
            ),
        )

    def _build_all_required_splits(
        self,
    ) -> dict[int, YearSplit]:

        required_years = sorted(
            set(
                self.years.final_source
            )
            | set(
                self.years.real_ood
            )
        )

        return {
            year: self._build_split(
                year
            )
            for year in required_years
        }

    def split(
        self,
        year: int,
    ) -> YearSplit:

        year = int(
            year
        )

        try:
            return self._splits[
                year
            ]

        except KeyError as exc:
            raise KeyError(
                f"Year {year} is outside the "
                "configured V3 protocol."
            ) from exc

    def verify_split_disjointness(
        self,
    ) -> None:
        """
        Fail hard if train / validation / test overlap or do
        not reconstruct the complete valid year.
        """

        for year, split in (
            self._splits.items()
        ):

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
                    f"Year {year}: train and validation overlap."
                )

            if train & test:
                raise RuntimeError(
                    f"Year {year}: train and test overlap."
                )

            if validation & test:
                raise RuntimeError(
                    f"Year {year}: validation and test overlap."
                )

            expected = set(
                range(
                    self.bundle.year(
                        year
                    ).n_samples
                )
            )

            reconstructed = (
                train
                | validation
                | test
            )

            if reconstructed != expected:
                missing = sorted(
                    expected
                    - reconstructed
                )

                extra = sorted(
                    reconstructed
                    - expected
                )

                raise RuntimeError(
                    f"Year {year}: split does not reconstruct "
                    "the complete year. "
                    f"missing={missing[:10]}, "
                    f"extra={extra[:10]}"
                )

    def supervised_slice(
        self,
        year: int,
    ) -> SupervisedSlice:

        year = int(
            year
        )

        data = self.bundle.year(
            year
        )

        split = self.split(
            year
        )

        if len(
            split.train
        ) == 0:
            raise RuntimeError(
                f"Year {year} has empty train split."
            )

        if len(
            split.validation
        ) == 0:
            raise RuntimeError(
                f"Year {year} has empty validation split."
            )

        return SupervisedSlice(
            year=year,

            X_train=data.X[
                split.train
            ],

            y_main_train=data.y_main[
                split.train
            ],

            y_aux_train=data.y_aux[
                split.train
            ],

            X_validation=data.X[
                split.validation
            ],

            y_main_validation=data.y_main[
                split.validation
            ],

            y_aux_validation=data.y_aux[
                split.validation
            ],
        )

    def stream_slice(
        self,
        *,
        year: int,
        split_name: SplitName | str,
    ) -> StreamSlice:

        year = int(
            year
        )

        split_name = SplitName(
            split_name
        )

        data = self.bundle.year(
            year
        )

        indices = self.split(
            year
        ).indices(
            split_name
        )

        if len(indices) == 0:
            raise RuntimeError(
                f"Year {year} has empty "
                f"{split_name.value} split."
            )

        # The split is randomly selected, but evaluation must
        # preserve the original relative order in the annual
        # CSV / embedding stream.
        chronological_indices = np.sort(
            indices
        )

        return StreamSlice(
            year=year,
            split_name=split_name,

            X=data.X[
                chronological_indices
            ],

            y_main=data.y_main[
                chronological_indices
            ],

            y_aux=data.y_aux[
                chronological_indices
            ],
        )

    def pseudo_source_supervised(
        self,
    ) -> tuple[SupervisedSlice, ...]:

        return tuple(
            self.supervised_slice(
                year
            )
            for year
            in self.years.pseudo_source
        )

    def pseudo_ood_validation_stream(
        self,
    ) -> tuple[StreamSlice, ...]:
        """
        TTA tuning stream.

        CRITICAL:
            validation split only.

        The 20% test split is intentionally inaccessible
        through this convenience method.
        """

        return tuple(
            self.stream_slice(
                year=year,
                split_name=(
                    SplitName.VALIDATION
                ),
            )
            for year
            in self.years.pseudo_ood
        )

    def final_source_supervised(
        self,
    ) -> tuple[SupervisedSlice, ...]:

        return tuple(
            self.supervised_slice(
                year
            )
            for year
            in self.years.final_source
        )

    def real_ood_test_stream(
        self,
    ) -> tuple[StreamSlice, ...]:
        """
        Final OOD evaluation stream.

        Accuracy is evaluated exclusively on the 20% test split.
        """

        return tuple(
            self.stream_slice(
                year=year,
                split_name=(
                    SplitName.TEST
                ),
            )
            for year
            in self.years.real_ood
        )

    def real_ood_supervised_reference(
        self,
    ) -> tuple[SupervisedSlice, ...]:
        """
        Per-year oracle/reference training data.

        Uses:
            70% train
            10% validation

        Its corresponding evaluation must use real_ood_test_stream().
        """

        return tuple(
            self.supervised_slice(
                year
            )
            for year
            in self.years.real_ood
        )

    def evaluator_stream(
        self,
        slices: Iterable[
            StreamSlice
        ],
    ) -> list[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ]:

        return [
            stream_slice.as_evaluator_tuple()
            for stream_slice
            in slices
        ]