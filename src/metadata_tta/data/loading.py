from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

MAIN_TARGET_COLUMN = "target"
AUX_TARGET_COLUMN = "aux_target"


# ============================================================
# YEAR FILE DISCOVERY
# ============================================================

def list_year_files(
    data_root: str | Path,
) -> list[tuple[int, Path]]:
    """
    Return annual CSV files sorted by year.

    Valid filenames are of the form:

        2002.csv
        2003.csv
        ...
    """

    root = Path(
        data_root
    ).expanduser()

    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}"
        )

    year_files: list[
        tuple[int, Path]
    ] = []

    for path in root.iterdir():

        if not path.is_file():
            continue

        if path.suffix.lower() != ".csv":
            continue

        match = re.fullmatch(
            r"(\d{4})",
            path.stem,
        )

        if match is None:
            continue

        year = int(
            match.group(1)
        )

        year_files.append(
            (
                year,
                path,
            )
        )

    year_files.sort(
        key=lambda item: item[0]
    )

    if not year_files:
        raise FileNotFoundError(
            "No annual CSV files found in "
            f"{root}. Expected filenames such as "
            "'2002.csv'."
        )

    return year_files


# ============================================================
# CSV LOADING
# ============================================================

def read_year_csv(
    path: str | Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Load one annual embedding CSV.

    Required columns
    ----------------
    target
        Main-task label.

    aux_target
        Metadata / auxiliary-task label.

    All remaining columns must contain numeric features.

    Row order is preserved exactly as stored in the CSV.
    """

    csv_path = Path(
        path
    ).expanduser()

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    frame = pd.read_csv(
        csv_path
    )

    required_columns = {
        MAIN_TARGET_COLUMN,
        AUX_TARGET_COLUMN,
    }

    missing = (
        required_columns
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{csv_path.name} is missing required "
            "columns: "
            + ", ".join(sorted(missing))
        )

    feature_columns = [
        column
        for column in frame.columns
        if column
        not in {
            MAIN_TARGET_COLUMN,
            AUX_TARGET_COLUMN,
        }
    ]

    if not feature_columns:
        raise ValueError(
            f"{csv_path.name} contains no feature columns."
        )

    # Convert features explicitly to numeric.
    #
    # If an accidental non-feature column such as "path"
    # is present, fail loudly instead of silently feeding
    # invalid data to the model.
    numeric_features = frame[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    entirely_non_numeric = [
        column
        for column in feature_columns
        if (
            numeric_features[
                column
            ]
            .notna()
            .sum()
            == 0
        )
    ]

    if entirely_non_numeric:
        raise ValueError(
            f"{csv_path.name} contains non-numeric "
            "columns that are not target columns: "
            + ", ".join(
                entirely_non_numeric
            )
        )

    X = numeric_features.to_numpy(
        dtype=np.float32,
        copy=True,
    )

    y_main = pd.to_numeric(
        frame[
            MAIN_TARGET_COLUMN
        ],
        errors="coerce",
    ).to_numpy()

    y_aux = pd.to_numeric(
        frame[
            AUX_TARGET_COLUMN
        ],
        errors="coerce",
    ).to_numpy()

    if not (
        len(X)
        == len(y_main)
        == len(y_aux)
    ):
        raise RuntimeError(
            "Feature and label arrays have "
            "inconsistent lengths."
        )

    return (
        X,
        y_main,
        y_aux,
    )


# ============================================================
# SAMPLE FILTERING
# ============================================================

def filter_valid_samples(
    X: np.ndarray,
    y_main: np.ndarray,
    y_aux: np.ndarray,
    n_classes_main: int,
    n_classes_aux: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Remove invalid samples.

    A sample is valid when:

    - every feature is finite;
    - main label is finite and integer-valued;
    - auxiliary label is finite and integer-valued;
    - main label lies in [0, n_classes_main);
    - aux label lies in [0, n_classes_aux).

    Relative row order is preserved.
    """

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y_main = np.asarray(
        y_main
    )

    y_aux = np.asarray(
        y_aux
    )

    if X.ndim != 2:
        raise ValueError(
            f"X must be 2-D, got shape {X.shape}."
        )

    if y_main.ndim != 1:
        y_main = y_main.reshape(-1)

    if y_aux.ndim != 1:
        y_aux = y_aux.reshape(-1)

    if not (
        len(X)
        == len(y_main)
        == len(y_aux)
    ):
        raise ValueError(
            "X, y_main and y_aux must have "
            "the same number of samples."
        )

    finite_features = np.all(
        np.isfinite(X),
        axis=1,
    )

    finite_main = np.isfinite(
        y_main
    )

    finite_aux = np.isfinite(
        y_aux
    )

    valid = (
        finite_features
        & finite_main
        & finite_aux
    )

    X = X[
        valid
    ]

    y_main = y_main[
        valid
    ]

    y_aux = y_aux[
        valid
    ]

    if len(y_main) == 0:
        return (
            X.astype(
                np.float32,
                copy=False,
            ),
            np.empty(
                0,
                dtype=np.int64,
            ),
            np.empty(
                0,
                dtype=np.int64,
            ),
        )

    integer_main = np.equal(
        y_main,
        np.floor(
            y_main
        ),
    )

    integer_aux = np.equal(
        y_aux,
        np.floor(
            y_aux
        ),
    )

    valid_integer = (
        integer_main
        & integer_aux
    )

    X = X[
        valid_integer
    ]

    y_main = y_main[
        valid_integer
    ].astype(
        np.int64,
        copy=False,
    )

    y_aux = y_aux[
        valid_integer
    ].astype(
        np.int64,
        copy=False,
    )

    valid_range = (
        (y_main >= 0)
        & (y_main < int(n_classes_main))
        & (y_aux >= 0)
        & (y_aux < int(n_classes_aux))
    )

    return (
        X[
            valid_range
        ].astype(
            np.float32,
            copy=False,
        ),
        y_main[
            valid_range
        ],
        y_aux[
            valid_range
        ],
    )