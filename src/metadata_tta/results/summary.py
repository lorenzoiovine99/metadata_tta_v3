from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from metadata_tta.evaluation import (
    EvaluationRecord,
    EvaluationResult,
)


def _statistics(
    records: Iterable[
        EvaluationRecord
    ],
) -> dict[str, Any]:

    records = tuple(
        records
    )

    if not records:
        return {
            "n_years": 0,
            "mean_accuracy": None,
            "std_accuracy": None,
            "worst_accuracy": None,
            "best_accuracy": None,
        }

    values = np.asarray(
        [
            float(record.accuracy)
            for record in records
        ],
        dtype=np.float64,
    )

    return {
        "n_years": len(records),
        "mean_accuracy": float(
            np.mean(values)
        ),
        "std_accuracy": float(
            np.std(values)
        ),
        "worst_accuracy": float(
            np.min(values)
        ),
        "best_accuracy": float(
            np.max(values)
        ),
    }


def build_final_summary(
    *,
    frozen_records: Iterable[
        EvaluationRecord
    ],
    tta_results: dict[
        str,
        EvaluationResult,
    ],
    supervised_reference: Iterable[
        EvaluationRecord
    ] = (),
) -> list[
    dict[str, Any]
]:
    """
    Build one compact final OOD summary.

    supervised_reference is intentionally labelled as
    'reference', not as a TTA method.
    """

    frozen_records = tuple(
        frozen_records
    )

    frozen_stats = _statistics(
        frozen_records
    )

    frozen_mean = (
        frozen_stats[
            "mean_accuracy"
        ]
    )

    if frozen_mean is None:
        raise RuntimeError(
            "Cannot build final summary without "
            "frozen OOD results."
        )

    rows: list[
        dict[str, Any]
    ] = []

    rows.append(
        {
            "method": "frozen",
            "family": "baseline",
            **frozen_stats,
            "delta_vs_frozen": 0.0,
        }
    )

    for method_name, result in (
        tta_results.items()
    ):

        stats = _statistics(
            result.records
        )

        mean_accuracy = stats[
            "mean_accuracy"
        ]

        rows.append(
            {
                "method": method_name,
                "family": "tta",
                **stats,
                "delta_vs_frozen": (
                    None
                    if mean_accuracy is None
                    else float(
                        mean_accuracy
                        - frozen_mean
                    )
                ),
            }
        )

    supervised_reference = tuple(
        supervised_reference
    )

    if supervised_reference:

        stats = _statistics(
            supervised_reference
        )

        mean_accuracy = stats[
            "mean_accuracy"
        ]

        rows.append(
            {
                "method":
                    "supervised_reference",
                "family":
                    "oracle_reference",
                **stats,
                "delta_vs_frozen": (
                    None
                    if mean_accuracy is None
                    else float(
                        mean_accuracy
                        - frozen_mean
                    )
                ),
            }
        )

    return rows