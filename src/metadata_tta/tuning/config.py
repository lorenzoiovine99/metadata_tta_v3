from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

from metadata_tta.config import (
    ExperimentConfig,
    load_config,
)


def load_tuning_yaml(
    path: str | Path,
) -> dict[str, Any]:

    tuning_path = Path(
        path
    ).expanduser().resolve()

    if not tuning_path.is_file():
        raise FileNotFoundError(
            f"Tuning config not found: "
            f"{tuning_path}"
        )

    with tuning_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw = yaml.safe_load(
            file
        )

    if not isinstance(raw, dict):
        raise ValueError(
            "Tuning config root must "
            "be a mapping."
        )

    return raw


def resolve_base_config(
    tuning_yaml_path: str | Path,
    tuning_config: Mapping[str, Any],
) -> tuple[
    Path,
    ExperimentConfig,
]:

    tuning_yaml_path = Path(
        tuning_yaml_path
    ).expanduser().resolve()

    raw_path = tuning_config.get(
        "base_config"
    )

    if (
        not isinstance(raw_path, str)
        or not raw_path.strip()
    ):
        raise ValueError(
            "Tuning config requires "
            "'base_config'."
        )

    candidate = Path(
        raw_path
    ).expanduser()

    if not candidate.is_absolute():
        # First interpret relative to repository cwd.
        cwd_candidate = (
            Path.cwd()
            / candidate
        )

        if cwd_candidate.is_file():
            candidate = cwd_candidate
        else:
            # Fallback relative to tuning YAML.
            candidate = (
                tuning_yaml_path.parent
                / candidate
            )

    candidate = (
        candidate.resolve()
    )

    return (
        candidate,
        load_config(
            candidate
        ),
    )


def set_nested_value(
    data: dict[str, Any],
    dotted_path: str,
    value: Any,
) -> None:
    """
    Set a dot-separated config path.

    Unlike ExperimentConfig.with_overrides(), this function
    intentionally allows creation of new optional mappings.
    This is needed for per-head overrides such as:

        model.single_head.dropout
        training.double_head.learning_rate
    """

    keys = dotted_path.split(
        "."
    )

    if not keys:
        raise ValueError(
            "Empty override path."
        )

    current = data

    for key in keys[:-1]:

        if key not in current:
            current[key] = {}

        next_value = current[key]

        if not isinstance(
            next_value,
            dict,
        ):
            raise ValueError(
                "Cannot descend through "
                f"{dotted_path!r}: "
                f"{key!r} is not a mapping."
            )

        current = next_value

    current[
        keys[-1]
    ] = copy.deepcopy(
        value
    )


def apply_overrides(
    config: ExperimentConfig,
    overrides: Mapping[str, Any],
) -> ExperimentConfig:

    data = config.as_dict()

    for path, value in (
        overrides.items()
    ):
        set_nested_value(
            data=data,
            dotted_path=str(path),
            value=value,
        )

    return ExperimentConfig(
        data
    )


def apply_overrides_to_dict(
    data: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:

    result = copy.deepcopy(
        dict(data)
    )

    for path, value in (
        overrides.items()
    ):
        set_nested_value(
            data=result,
            dotted_path=str(path),
            value=value,
        )

    return result