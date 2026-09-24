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
class SourceStreamYear:
    year: int

    # Supervised training split
    X_supervised: np.ndarray
    y_main_supervised: np.ndarray
    y_aux_supervised: np.ndarray

    # Validation split used only for early stopping
    X_validation: np.ndarray
    y_main_validation: np.ndarray
    y_aux_validation: np.ndarray

    # Held-out ID split used only for evaluation
    X_id: np.ndarray
    y_main_id: np.ndarray
    y_aux_id: np.ndarray


@dataclass(frozen=True)
class OODStreamYear:
    year: int

    # Supervised yearly TAS reference training split
    X_reference_train: np.ndarray
    y_main_reference_train: np.ndarray
    y_aux_reference_train: np.ndarray

    # Validation split for early stopping of the yearly
    # supervised TAS reference model
    X_reference_validation: np.ndarray
    y_main_reference_validation: np.ndarray
    y_aux_reference_validation: np.ndarray

    # Held-out chronological OOD test split
    X_test: np.ndarray
    y_main_test: np.ndarray
    y_aux_test: np.ndarray


@dataclass(frozen=True)
class EvalStreamTASData:
    source_years: tuple[
        SourceStreamYear,
        ...
    ]

    ood_years: tuple[
        OODStreamYear,
        ...
    ]


def _get_year(
    bundle: DataBundle,
    year: int,
) -> YearData:

    return bundle.year(
        int(
            year
        )
    )


def build_eval_stream_tas(
    bundle: DataBundle,
    config: ExperimentConfig,
) -> EvalStreamTASData:
    """
    Eval-Stream-TAS protocol.

    Every year is split into:
        train
        validation
        test

    Source years:
        train:
            supervised training / yearly fine-tuning.

        validation:
            early stopping only.

        test:
            held-out yearly ID evaluation.

    OOD years:
        train:
            supervised TAS reference fine-tuning.

        validation:
            early stopping for supervised TAS references.

        test:
            chronological OOD evaluation stream.

    The test split is never used for training or early stopping.

    Test streams preserve original CSV row order by sorting
    the selected test indices.
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

    # ========================================================
    # SOURCE YEARS
    # ========================================================

    source_years: list[
        SourceStreamYear
    ] = []

    for year in range(
        source_start,
        source_end + 1,
    ):

        data = _get_year(
            bundle=bundle,
            year=year,
        )

        split = build_split_indices(
            y_main=data.y_main,
            test_size=test_size,
            validation_size_within_train=(
                validation_size
            ),
            seed=split_seed,
        )

        train_indices = np.asarray(
            split.train
        )

        validation_indices = np.asarray(
            split.validation
        )

        id_indices = np.sort(
            split.test
        )

        if len(
            validation_indices
        ) == 0:
            raise ValueError(
                f"Source year {year} has an empty "
                "validation split. Early stopping "
                "requires validation samples."
            )

        source_years.append(
            SourceStreamYear(
                year=year,

                # --------------------------------------------
                # TRAIN
                # --------------------------------------------

                X_supervised=data.X[
                    train_indices
                ],

                y_main_supervised=data.y_main[
                    train_indices
                ],

                y_aux_supervised=data.y_aux[
                    train_indices
                ],

                # --------------------------------------------
                # VALIDATION
                # --------------------------------------------

                X_validation=data.X[
                    validation_indices
                ],

                y_main_validation=data.y_main[
                    validation_indices
                ],

                y_aux_validation=data.y_aux[
                    validation_indices
                ],

                # --------------------------------------------
                # TEST / ID
                # --------------------------------------------

                X_id=data.X[
                    id_indices
                ],

                y_main_id=data.y_main[
                    id_indices
                ],

                y_aux_id=data.y_aux[
                    id_indices
                ],
            )
        )

    # ========================================================
    # OOD YEARS
    # ========================================================

    ood_years: list[
        OODStreamYear
    ] = []

    for year in range(
        ood_start,
        ood_end + 1,
    ):

        data = _get_year(
            bundle=bundle,
            year=year,
        )

        split = build_split_indices(
            y_main=data.y_main,
            test_size=test_size,
            validation_size_within_train=(
                validation_size
            ),
            seed=split_seed,
        )

        train_indices = np.asarray(
            split.train
        )

        validation_indices = np.asarray(
            split.validation
        )

        test_indices = np.sort(
            split.test
        )

        if len(
            validation_indices
        ) == 0:
            raise ValueError(
                f"OOD year {year} has an empty "
                "validation split. Early stopping "
                "requires validation samples."
            )

        ood_years.append(
            OODStreamYear(
                year=year,

                # --------------------------------------------
                # SUPERVISED REFERENCE TRAIN
                # --------------------------------------------

                X_reference_train=data.X[
                    train_indices
                ],

                y_main_reference_train=data.y_main[
                    train_indices
                ],

                y_aux_reference_train=data.y_aux[
                    train_indices
                ],

                # --------------------------------------------
                # SUPERVISED REFERENCE VALIDATION
                # --------------------------------------------

                X_reference_validation=data.X[
                    validation_indices
                ],

                y_main_reference_validation=data.y_main[
                    validation_indices
                ],

                y_aux_reference_validation=data.y_aux[
                    validation_indices
                ],

                # --------------------------------------------
                # HELD-OUT OOD TEST
                # --------------------------------------------

                X_test=data.X[
                    test_indices
                ],

                y_main_test=data.y_main[
                    test_indices
                ],

                y_aux_test=data.y_aux[
                    test_indices
                ],
            )
        )

    return EvalStreamTASData(
        source_years=tuple(
            source_years
        ),
        ood_years=tuple(
            ood_years
        ),
    )