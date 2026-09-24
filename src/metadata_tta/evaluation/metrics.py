from __future__ import annotations

import numpy as np


def accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have identical shapes."
        )

    if y_true.size == 0:
        raise ValueError(
            "Cannot compute accuracy on zero samples."
        )

    return float(
        np.mean(
            y_true == y_pred
        )
    )