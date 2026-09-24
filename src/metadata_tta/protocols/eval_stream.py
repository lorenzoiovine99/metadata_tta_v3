from __future__ import annotations

from dataclasses import dataclass

from metadata_tta.config import ExperimentConfig
from metadata_tta.data import DataBundle

from .eval_fix import (
    EvaluationYear,
    _get_year,
)


@dataclass(frozen=True)
class EvalStreamData:
    evaluation_years: tuple[
        EvaluationYear,
        ...
    ]
    ood_horizon: int


def build_eval_stream(
    bundle: DataBundle,
    config: ExperimentConfig,
) -> EvalStreamData:

    protocol = config.section(
        "protocol"
    )

    start_year = int(
        protocol[
            "evaluation_start_year"
        ]
    )

    end_year = int(
        protocol[
            "evaluation_end_year"
        ]
    )

    ood_horizon = int(
        protocol[
            "ood_horizon"
        ]
    )

    years: list[
        EvaluationYear
    ] = []

    for year in range(
        start_year,
        end_year + 1,
    ):

        data = _get_year(
            bundle,
            year,
        )

        years.append(
            EvaluationYear(
                year=year,
                X=data.X,
                y_main=data.y_main,
                y_aux=data.y_aux,
            )
        )

    return EvalStreamData(
        evaluation_years=tuple(
            years
        ),
        ood_horizon=ood_horizon,
    )