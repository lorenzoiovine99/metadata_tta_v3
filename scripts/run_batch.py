from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run multiple Metadata TTA V3 experiments "
            "serially."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Batch YAML, e.g. "
            "configs/batches/fmow_5seeds.yaml"
        ),
    )

    return parser.parse_args()


def _load_batch_config(
    path: Path,
) -> dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(
            handle
        )

    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            "Batch config must contain a mapping."
        )

    required = {
        "dataset",
        "mode",
        "seeds",
    }

    missing = (
        required
        - set(config)
    )

    if missing:
        raise ValueError(
            "Batch config missing fields: "
            f"{sorted(missing)}"
        )

    if config["mode"] not in {
        "tune",
        "test",
        "tune_and_test",
    }:
        raise ValueError(
            "Invalid batch mode: "
            f"{config['mode']!r}"
        )

    seeds = config["seeds"]

    if (
        not isinstance(seeds, list)
        or not seeds
    ):
        raise ValueError(
            "seeds must be a non-empty list "
            "of [experiment_seed, split_seed] pairs."
        )

    parsed_seeds: list[
        tuple[int, int]
    ] = []

    for row_index, row in enumerate(
        seeds
    ):

        if (
            not isinstance(row, list)
            or len(row) != 2
        ):
            raise ValueError(
                f"Invalid seeds row {row_index}: "
                f"{row!r}. Expected "
                "[experiment_seed, split_seed]."
            )

        experiment_seed = int(
            row[0]
        )

        split_seed = int(
            row[1]
        )

        parsed_seeds.append(
            (
                experiment_seed,
                split_seed,
            )
        )

    config["seeds"] = (
        parsed_seeds
    )

    return config


def _resolve_path(
    value: str | Path,
) -> Path:

    path = Path(
        value
    )

    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            / path
        )

    return path.resolve()


def main() -> None:

    args = _parse_args()

    batch_path = _resolve_path(
        args.config
    )

    batch = _load_batch_config(
        batch_path
    )

    dataset_name = str(
        batch["dataset"]
    )

    mode = str(
        batch["mode"]
    )

    seeds: list[
        tuple[int, int]
    ] = list(
        batch["seeds"]
    )

    continue_on_error = bool(
        batch.get(
            "continue_on_error",
            False,
        )
    )

    dataset_config = _resolve_path(
        batch.get(
            "dataset_config",
            (
                f"configs/datasets/"
                f"{dataset_name}.yaml"
            ),
        )
    )

    experiment_config = _resolve_path(
        batch.get(
            "experiment_config",
            "configs/experiments/default.yaml",
        )
    )

    tuning_config = _resolve_path(
        batch.get(
            "tuning_config",
            "configs/tuning/default.yaml",
        )
    )

    best_overrides = batch.get(
        "best_overrides"
    )

    if (
        mode == "test"
        and best_overrides is not None
    ):
        best_overrides = _resolve_path(
            best_overrides
        )

    if not dataset_config.exists():
        raise FileNotFoundError(
            dataset_config
        )

    if not experiment_config.exists():
        raise FileNotFoundError(
            experiment_config
        )

    if (
        mode in {
            "tune",
            "tune_and_test",
        }
        and not tuning_config.exists()
    ):
        raise FileNotFoundError(
            tuning_config
        )

    run_script = (
        PROJECT_ROOT
        / "scripts"
        / "run.py"
    )

    print()
    print("=" * 80)
    print("METADATA TTA V3 — BATCH")
    print("=" * 80)
    print(
        f"dataset={dataset_name}"
    )
    print(
        f"mode={mode}"
    )
    print(
        f"seeds={seeds}"
    )
    print(
        f"n_runs={len(seeds)}"
    )
    print(
        f"continue_on_error="
        f"{continue_on_error}"
    )
    print("=" * 80)

    failures: list[
        tuple[int, int, int]
    ] = []

    total = len(
        seeds
    )

    for index, (
        experiment_seed,
        split_seed,
    ) in enumerate(
        seeds,
        start=1,
    ):

        print()
        print("=" * 80)
        print(
            f"BATCH RUN {index}/{total}"
        )
        print(
            f"experiment_seed="
            f"{experiment_seed}"
        )
        print(
            f"split_seed="
            f"{split_seed}"
        )
        print("=" * 80)

        command = [
            sys.executable,
            str(run_script),
            "--dataset-config",
            str(dataset_config),
            "--experiment-config",
            str(experiment_config),
            "--mode",
            mode,
            "--experiment-seed",
            str(experiment_seed),
            "--split-seed",
            str(split_seed),
        ]

        if mode in {
            "tune",
            "tune_and_test",
        }:
            command.extend(
                [
                    "--tuning-config",
                    str(tuning_config),
                ]
            )

        if (
            mode == "test"
            and best_overrides is not None
        ):
            command.extend(
                [
                    "--best-overrides",
                    str(best_overrides),
                ]
            )

        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
        )

        if completed.returncode == 0:

            print()
            print(
                "[BATCH][PASS] "
                f"experiment_seed={experiment_seed} | "
                f"split_seed={split_seed}"
            )

            continue

        failures.append(
            (
                experiment_seed,
                split_seed,
                completed.returncode,
            )
        )

        print()
        print(
            "[BATCH][FAIL] "
            f"experiment_seed={experiment_seed} | "
            f"split_seed={split_seed} | "
            f"returncode={completed.returncode}"
        )

        if not continue_on_error:
            raise SystemExit(
                completed.returncode
            )

    print()
    print("=" * 80)
    print("BATCH COMPLETE")
    print("=" * 80)

    if failures:

        print(
            f"successful="
            f"{total - len(failures)}/{total}"
        )

        print(
            f"failed="
            f"{len(failures)}/{total}"
        )

        for (
            experiment_seed,
            split_seed,
            returncode,
        ) in failures:

            print(
                "[FAIL] "
                f"experiment_seed={experiment_seed} | "
                f"split_seed={split_seed} | "
                f"returncode={returncode}"
            )

        raise SystemExit(
            1
        )

    print(
        f"[CHECK][PASS] "
        f"all {total} runs completed"
    )


if __name__ == "__main__":
    main()