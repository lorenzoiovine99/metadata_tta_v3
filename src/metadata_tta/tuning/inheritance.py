from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .results import load_best_result


def inherited_overrides_from_best(
    *,
    best_path: str | Path,
    parameter_mapping: Mapping[
        str,
        str | Sequence[str],
    ],
) -> dict[str, Any]:
    """
    Read logical parameter values from a parent's best.yaml
    and map them onto config paths used by the child stage.

    Example
    -------
    Parent single-head best.yaml:

        parameters:
          learning_rate: 0.001
          dropout: 0.1

    Child mapping:

        learning_rate:
          - training.double_head.learning_rate

        dropout:
          - model.double_head.dropout

    Result:

        {
            "training.double_head.learning_rate": 0.001,
            "model.double_head.dropout": 0.1,
        }

    The inheritance therefore happens by logical parameter
    name, not by blindly copying parent config paths.
    """

    result = load_best_result(
        best_path
    )

    parameters = result.get(
        "parameters",
        {},
    )

    if not isinstance(
        parameters,
        Mapping,
    ):
        raise ValueError(
            f"Invalid parameters section in {best_path}"
        )

    overrides: dict[
        str,
        Any,
    ] = {}

    for (
        parameter_name,
        destination_paths,
    ) in parameter_mapping.items():

        if parameter_name not in parameters:
            raise KeyError(
                f"Parent tuning result {best_path} "
                f"does not contain parameter "
                f"{parameter_name!r}."
            )

        value = parameters[
            parameter_name
        ]

        if isinstance(
            destination_paths,
            str,
        ):
            paths = [
                destination_paths
            ]
        else:
            paths = list(
                destination_paths
            )

        if not paths:
            raise ValueError(
                f"Inheritance mapping for "
                f"{parameter_name!r} is empty."
            )

        for path in paths:
            overrides[
                str(path)
            ] = value

    return overrides


def merge_overrides(
    *groups: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Merge override dictionaries from left to right.

    Later groups take precedence.

    This is deliberate:
        parent inherited values
            ↓
        current trial values

    Therefore a child trial can refine an inherited value
    simply by overriding the same config path.
    """

    merged: dict[
        str,
        Any,
    ] = {}

    for group in groups:
        merged.update(
            {
                str(key): value
                for key, value
                in group.items()
            }
        )

    return merged