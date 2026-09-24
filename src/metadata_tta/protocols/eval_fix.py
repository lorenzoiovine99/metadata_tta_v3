from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metadata_tta.config import ExperimentConfig
from metadata_tta.data import (
    DataBundle,
    YearData,
    build_split_indices,
)


@dataclass(frozen=True)
class SourceYear:
    year: int
    X_supervised: np.ndarray
    y_main_supervised: np.ndarray
    y_aux_supervised: np.ndarray


@dataclass(frozen=True)
class EvaluationYear:
    year: int
    X: np.ndarray
    y_main: np.ndarray
    y_aux: np.ndarray


@dataclass(frozen=True)
class EvalFixData:
    source_years: tuple[SourceYear, ...]
    id_year: EvaluationYear
    ood_years: tuple[EvaluationYear, ...]

def _get_year(
    bundle: DataBundle,
    year: int,
) -> YearData:
    return bundle.year(
        int(year)
    )

def build_eval_fix(
    bundle: DataBundle,
    config: ExperimentConfig,
) -> EvalFixData:
    """
    Eval-Fix protocol.

    Source:
        source_start_year ... source_end_year

        For every source year:
            80% supervised = train + validation
            20% held-out

    ID:
        held-out 20% of source_end_year only

    OOD:
        100% of every OOD year in original row order.

    No OOD split, shuffle or subsampling is performed.
    """

    protocol = config.section(
        "protocol"
    )

    dataset = config.section(
        "dataset"
    )

    split_config = dataset[
        "split"
    ]

    source_start = int(
        protocol[
            "source_start_year"
        ]
    )

    source_end = int(
        protocol[
            "source_end_year"
        ]
    )

    ood_start = int(
        protocol[
            "ood_start_year"
        ]
    )

    ood_end = int(
        protocol[
            "ood_end_year"
        ]
    )

    test_size = float(
        split_config[
            "test_size"
        ]
    )

    validation_size = float(
        split_config[
            "validation_size_within_train"
        ]
    )

    split_seed = int(
        split_config[
            "seed"
        ]
    )

    source_years: list[
        SourceYear
    ] = []

    cutoff_test_indices: (
        np.ndarray | None
    ) = None

    # ========================================================
    # SOURCE YEARS
    # ========================================================

    for year in range(
        source_start,
        source_end + 1,
    ):

        data = _get_year(
            bundle,
            year,
        )

        split = build_split_indices(
            y_main=data.y_main,
            test_size=test_size,
            validation_size_within_train=validation_size,
            seed=split_seed,
        )

        supervised_indices = np.concatenate(
            [
                split.train,
                split.validation,
            ]
        )

        source_years.append(
            SourceYear(
                year=year,
                X_supervised=data.X[
                    supervised_indices
                ],
                y_main_supervised=data.y_main[
                    supervised_indices
                ],
                y_aux_supervised=data.y_aux[
                    supervised_indices
                ],
            )
        )

        if year == source_end:
            cutoff_test_indices = (
                split.test.copy()
            )

    if cutoff_test_indices is None:
        raise RuntimeError(
            "Failed to construct cutoff-year "
            "held-out ID split."
        )

    # ========================================================
    # ID = HELD-OUT 20% OF CUTOFF YEAR
    # ========================================================

    cutoff_data = _get_year(
        bundle,
        source_end,
    )

    id_year = EvaluationYear(
        year=source_end,
        X=cutoff_data.X[
            cutoff_test_indices
        ],
        y_main=cutoff_data.y_main[
            cutoff_test_indices
        ],
        y_aux=cutoff_data.y_aux[
            cutoff_test_indices
        ],
    )

    # ========================================================
    # OOD = 100% IN ORIGINAL CSV ORDER
    # ========================================================

    ood_years: list[
        EvaluationYear
    ] = []

    for year in range(
        ood_start,
        ood_end + 1,
    ):

        data = _get_year(
            bundle,
            year,
        )

        ood_years.append(
            EvaluationYear(
                year=year,
                X=data.X,
                y_main=data.y_main,
                y_aux=data.y_aux,
            )
        )

    return EvalFixData(
        source_years=tuple(
            source_years
        ),
        id_year=id_year,
        ood_years=tuple(
            ood_years
        ),
    )