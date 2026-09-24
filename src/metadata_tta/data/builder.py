from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from metadata_tta.config import ExperimentConfig

from .loading import (
    filter_valid_samples,
    list_year_files,
    read_year_csv,
)


# ============================================================
# YEAR DATA
# ============================================================

@dataclass(frozen=True)
class YearData:
    """
    Canonical representation of one complete temporal slice.

    Samples remain in the same relative order as in the
    original CSV after invalid samples are removed.
    """

    year: int

    X: np.ndarray

    y_main: np.ndarray

    y_aux: np.ndarray

    @property
    def n_samples(
        self,
    ) -> int:

        return int(
            len(
                self.y_main
            )
        )

    @property
    def feature_dim(
        self,
    ) -> int:

        return int(
            self.X.shape[1]
        )


# ============================================================
# DATA BUNDLE
# ============================================================

@dataclass(frozen=True)
class DataBundle:
    """
    Complete canonical temporal dataset.

    Protocol-specific train/test construction happens later.
    """

    dataset_name: str

    years: tuple[int, ...]

    by_year: dict[int, YearData]

    feature_dim: int

    data_root: Path

    def year(
        self,
        year: int,
    ) -> YearData:

        try:
            return self.by_year[
                int(year)
            ]

        except KeyError as exc:
            raise KeyError(
                f"Year {year} not available in dataset "
                f"{self.dataset_name!r}."
            ) from exc


# ============================================================
# BUILDER
# ============================================================

def build_data_bundle(
    config: ExperimentConfig,
) -> DataBundle:
    """
    Load the complete temporal dataset.

    Important
    ---------
    This function does NOT:

    - create train/test splits;
    - decide source/OOD years;
    - fit PCA;
    - know whether the protocol is Eval-Fix or Eval-Stream.

    Its only job is to produce one clean canonical
    representation of every year.
    """

    dataset_config = config.section(
        "dataset"
    )

    data_root = config.data_root

    n_classes_main = int(
        dataset_config[
            "n_classes_main"
        ]
    )

    n_classes_aux = int(
        dataset_config[
            "n_classes_aux"
        ]
    )

    year_files = list_year_files(
        data_root
    )

    by_year: dict[
        int,
        YearData
    ] = {}

    feature_dim: int | None = None

    print()
    print(
        "=" * 80
    )

    print(
        "DATASET"
    )

    print(
        "=" * 80
    )

    print(
        "Dataset:",
        config.dataset_name,
    )

    print(
        "Data root:",
        data_root,
    )

    print()

    for year, path in year_files:

        (
            X,
            y_main,
            y_aux,
        ) = read_year_csv(
            path
        )

        n_raw = int(
            len(
                y_main
            )
        )

        (
            X,
            y_main,
            y_aux,
        ) = filter_valid_samples(
            X=X,
            y_main=y_main,
            y_aux=y_aux,
            n_classes_main=(
                n_classes_main
            ),
            n_classes_aux=(
                n_classes_aux
            ),
        )

        if len(X) == 0:
            print(
                f"Year {year}: "
                f"0 valid samples "
                f"(raw={n_raw}) -- skipped"
            )

            continue

        current_feature_dim = int(
            X.shape[1]
        )

        if feature_dim is None:
            feature_dim = (
                current_feature_dim
            )

        elif (
            current_feature_dim
            != feature_dim
        ):
            raise ValueError(
                f"Feature dimension mismatch in year {year}: "
                f"expected {feature_dim}, "
                f"found {current_feature_dim}."
            )

        by_year[
            int(year)
        ] = YearData(
            year=int(
                year
            ),
            X=np.asarray(
                X,
                dtype=np.float32,
            ),
            y_main=np.asarray(
                y_main,
                dtype=np.int64,
            ),
            y_aux=np.asarray(
                y_aux,
                dtype=np.int64,
            ),
        )

        print(
            f"Year {year}: "
            f"valid={len(y_main)}/{n_raw} | "
            f"main_classes={len(np.unique(y_main))} | "
            f"aux_classes={len(np.unique(y_aux))}"
        )

    if not by_year:
        raise ValueError(
            "No valid annual data were loaded."
        )

    if feature_dim is None:
        raise RuntimeError(
            "Unable to infer feature dimension."
        )

    years = tuple(
        sorted(
            by_year.keys()
        )
    )

    print()

    print(
        "Years:",
        f"{years[0]}-{years[-1]}",
    )

    print(
        "Number of years:",
        len(years),
    )

    print(
        "Feature dimension:",
        feature_dim,
    )

    print(
        "Total valid samples:",
        sum(
            year_data.n_samples
            for year_data
            in by_year.values()
        ),
    )

    return DataBundle(
        dataset_name=(
            config.dataset_name
        ),
        years=years,
        by_year=by_year,
        feature_dim=int(
            feature_dim
        ),
        data_root=Path(
            data_root
        ),
    )