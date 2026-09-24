from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from metadata_tta.config import (
    ExperimentConfig,
)

from .config import (
    apply_overrides_to_dict,
)
from .results import (
    load_best_result,
)


def _best_overrides(
    path: Path,
) -> dict[str, Any]:

    result = load_best_result(
        path
    )

    overrides = result.get(
        "overrides",
        {},
    )

    if not isinstance(
        overrides,
        Mapping,
    ):
        raise ValueError(
            f"Invalid tuning result: "
            f"{path}"
        )

    return dict(
        overrides
    )


def materialize_tuned_config(
    *,
    base_config: ExperimentConfig,
    output_root: Path,
    enabled_methods: list[str],
    destination: Path,
) -> Path:

    data = base_config.as_dict()

    # ---------------------------------------------------------
    # 1. Apply tuned SINGLE-HEAD configuration
    # ---------------------------------------------------------

    single_best_path = (
        output_root
        / "baselines"
        / "single_head"
        / "best.yaml"
    )

    if not single_best_path.is_file():
        raise FileNotFoundError(
            f"Missing single-head tuning result: {single_best_path}"
        )

    data = apply_overrides_to_dict(
        data=data,
        overrides=_best_overrides(single_best_path),
    )

    # ---------------------------------------------------------
    # 2. Keep DOUBLE architecture identical to SINGLE
    #
    # The double model is not independently tuned anymore.
    # Its main path must exactly match the tuned single model
    # so that single -> double weight copying is valid.
    # ---------------------------------------------------------

    data = apply_overrides_to_dict(
        data=data,
        overrides={
            "model.double_head.shared_hidden_dim": (
                data["model"]["single_head"]["shared_hidden_dim"]
            ),
            "model.double_head.dropout": (
                data["model"]["single_head"]["dropout"]
            ),
        },
    )

    # ---------------------------------------------------------
    # 3. Apply tuned TTA configurations
    # ---------------------------------------------------------

    for method_name in enabled_methods:

        path = (
            output_root
            / "tta"
            / method_name
            / "best.yaml"
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing TTA tuning result: {path}"
            )

        data = apply_overrides_to_dict(
            data=data,
            overrides=_best_overrides(path),
        )

    # ---------------------------------------------------------
    # 4. Write materialized config
    # ---------------------------------------------------------

    destination = Path(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with destination.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
        )

    return destination