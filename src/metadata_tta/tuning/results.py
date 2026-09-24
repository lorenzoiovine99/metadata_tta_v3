from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


def ensure_directory(
    path: str | Path,
) -> Path:

    directory = Path(
        path
    ).expanduser()

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


def save_trial_rows(
    path: str | Path,
    rows: list[dict[str, Any]],
) -> None:

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    fields: list[str] = []

    seen: set[str] = set()

    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(
                    key
                )
                seen.add(
                    key
                )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:

            serializable = {}

            for key, value in (
                row.items()
            ):

                if isinstance(
                    value,
                    (
                        dict,
                        list,
                        tuple,
                    ),
                ):
                    serializable[key] = (
                        json.dumps(
                            value,
                            sort_keys=True,
                        )
                    )
                else:
                    serializable[
                        key
                    ] = value

            writer.writerow(
                serializable
            )


def save_best_result(
    path: str | Path,
    *,
    kind: str,
    name: str,
    score_name: str,
    score: float,
    parameters: Mapping[str, Any],
    overrides: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> None:

    payload: dict[str, Any] = {
        "kind": kind,
        "name": name,
        "score_name": score_name,
        "score": float(score),
        "parameters": dict(
            parameters
        ),
        "overrides": dict(
            overrides
        ),
    }

    if extra:
        payload[
            "extra"
        ] = dict(
            extra
        )

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            payload,
            file,
            sort_keys=False,
        )


def load_best_result(
    path: str | Path,
) -> dict[str, Any]:

    path = Path(
        path
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = yaml.safe_load(
            file
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            f"Invalid best-result file: {path}"
        )

    return payload