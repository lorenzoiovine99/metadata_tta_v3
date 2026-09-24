from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.decomposition import IncrementalPCA


# ============================================================
# PCA FIT
# ============================================================

def fit_incremental_pca(
    arrays: Iterable[np.ndarray],
    n_components: int,
) -> IncrementalPCA:
    """
    Fit IncrementalPCA on a sequence of source-only arrays.

    No target/OOD data should ever be passed to this function.
    """

    n_components = int(
        n_components
    )

    if n_components < 1:
        raise ValueError(
            "n_components must be >= 1."
        )

    batch_size = max(
        2048,
        n_components * 4,
    )

    pca = IncrementalPCA(
        n_components=n_components,
        batch_size=batch_size,
    )

    buffered: list[
        np.ndarray
    ] = []

    buffered_rows = 0
    fitted = False

    for array in arrays:

        X = np.asarray(
            array,
            dtype=np.float32,
        )

        if X.ndim != 2:
            raise ValueError(
                "Every PCA input must be 2-D."
            )

        if len(X) == 0:
            continue

        if not fitted:

            buffered.append(
                X
            )

            buffered_rows += len(
                X
            )

            if buffered_rows < n_components:
                continue

            initial = np.concatenate(
                buffered,
                axis=0,
            )

            first_end = min(
                batch_size,
                len(initial),
            )

            pca.partial_fit(
                initial[
                    :first_end
                ]
            )

            fitted = True

            for start in range(
                first_end,
                len(initial),
                batch_size,
            ):

                end = min(
                    start + batch_size,
                    len(initial),
                )

                chunk = initial[
                    start:end
                ]

                if len(chunk) < n_components:
                    # IncrementalPCA requires at least
                    # n_components samples in each partial_fit.
                    break

                pca.partial_fit(
                    chunk
                )

            buffered = []
            buffered_rows = 0

        else:

            for start in range(
                0,
                len(X),
                batch_size,
            ):

                end = min(
                    start + batch_size,
                    len(X),
                )

                chunk = X[
                    start:end
                ]

                if len(chunk) < n_components:
                    continue

                pca.partial_fit(
                    chunk
                )

    if not fitted:
        raise ValueError(
            "Not enough source samples to fit PCA: "
            f"at least {n_components} are required."
        )

    return pca


# ============================================================
# PCA TRANSFORM
# ============================================================

def transform_features(
    X: np.ndarray,
    pca: IncrementalPCA | None,
) -> np.ndarray:
    """
    Apply PCA when available, otherwise return float32 features.
    """

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    if pca is None:
        return X.astype(
            np.float32,
            copy=False,
        )

    return pca.transform(
        X
    ).astype(
        np.float32,
        copy=False,
    )