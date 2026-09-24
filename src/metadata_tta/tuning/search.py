from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class SearchTrial:
    """
    One hyperparameter-search trial.

    parameters:
        Human-readable parameter values.

    overrides:
        Dot-separated ExperimentConfig overrides.
    """

    trial_id: int
    parameters: dict[str, Any]
    overrides: dict[str, Any]


class SearchStrategy(ABC):
    """
    Search-strategy abstraction.

    Grid search is implemented now. Optuna can later implement
    this same interface without changing baseline/TTA objectives.
    """

    @abstractmethod
    def generate(
        self,
        space: Mapping[str, Any],
    ) -> Iterator[SearchTrial]:
        raise NotImplementedError


def _choices_for_parameter(
    name: str,
    spec: Mapping[str, Any],
) -> list[tuple[Any, dict[str, Any]]]:

    if "values" in spec:
        values = list(spec["values"])

        paths = spec.get(
            "paths",
            [name],
        )

        if not paths:
            raise ValueError(
                f"Search parameter {name!r} has no paths."
            )

        result: list[
            tuple[Any, dict[str, Any]]
        ] = []

        for value in values:
            overrides = {
                str(path): value
                for path in paths
            }

            result.append(
                (
                    value,
                    overrides,
                )
            )

        return result

    if "value_overrides" in spec:
        entries = list(
            spec["value_overrides"]
        )

        result = []

        for index, entry in enumerate(entries):

            if not isinstance(entry, Mapping):
                raise ValueError(
                    f"{name}.value_overrides[{index}] "
                    "must be a mapping."
                )

            label = entry.get(
                "label",
                index,
            )

            overrides = entry.get(
                "overrides"
            )

            if not isinstance(
                overrides,
                Mapping,
            ):
                raise ValueError(
                    f"{name}.value_overrides[{index}]."
                    "overrides must be a mapping."
                )

            result.append(
                (
                    label,
                    dict(overrides),
                )
            )

        return result

    raise ValueError(
        f"Search parameter {name!r} must contain "
        "'values' or 'value_overrides'."
    )


class GridSearchStrategy(SearchStrategy):

    def generate(
        self,
        space: Mapping[str, Any],
    ) -> Iterator[SearchTrial]:

        names = list(
            space.keys()
        )

        if not names:
            yield SearchTrial(
                trial_id=0,
                parameters={},
                overrides={},
            )
            return

        choices = [
            _choices_for_parameter(
                name=name,
                spec=space[name],
            )
            for name in names
        ]

        for trial_id, combination in enumerate(
            product(*choices)
        ):
            parameters: dict[
                str,
                Any,
            ] = {}

            overrides: dict[
                str,
                Any,
            ] = {}

            for name, (
                display_value,
                parameter_overrides,
            ) in zip(
                names,
                combination,
                strict=True,
            ):
                parameters[name] = (
                    display_value
                )

                overrides.update(
                    parameter_overrides
                )

            yield SearchTrial(
                trial_id=trial_id,
                parameters=parameters,
                overrides=overrides,
            )


def build_search_strategy(
    name: str,
) -> SearchStrategy:

    normalized = str(
        name
    ).strip().lower()

    if normalized == "grid":
        return GridSearchStrategy()

    raise ValueError(
        f"Unknown search strategy {name!r}. "
        "Currently supported: grid."
    )