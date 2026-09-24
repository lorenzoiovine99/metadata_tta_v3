from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SRC_ROOT = (
    PROJECT_ROOT
    / "src"
)

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_ROOT),
    )


from metadata_tta.config import (
    compose_config,
    load_yaml,
)
from metadata_tta.experiment import (
    run_experiment,
)
from metadata_tta.results import (
    ResultsWriter,
    build_final_summary,
)
from metadata_tta.utils import (
    configure_logging,
)


def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Metadata-guided Test-Time Adaptation V3"
        )
    )

    parser.add_argument(
        "--dataset-config",
        required=True,
        help=(
            "Dataset YAML, e.g. "
            "configs/datasets/fmow.yaml"
        ),
    )

    parser.add_argument(
        "--experiment-config",
        default=(
            "configs/experiments/default.yaml"
        ),
    )

    parser.add_argument(
        "--tuning-config",
        default=(
            "configs/tuning/default.yaml"
        ),
    )

    parser.add_argument(
        "--mode",
        choices=[
            "tune",
            "test",
            "tune_and_test",
        ],
        default=None,
    )

    parser.add_argument(
        "--experiment-seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--best-overrides",
        default=None,
        help=(
            "Optional YAML/JSON mapping of dot-path "
            "overrides for mode=test."
        ),
    )

    return parser.parse_args()


def _git_commit() -> str | None:

    try:

        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except Exception:
        return None


def _load_mapping(
    path: str | Path,
) -> dict[str, Any]:

    path = Path(
        path
    ).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    if path.suffix.lower() == ".json":

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            payload = json.load(
                handle
            )

    else:

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            payload = yaml.safe_load(
                handle
            )

    if payload is None:
        return {}

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            f"{path} must contain a mapping."
        )

    return payload


def _run_directory(
    *,
    results_root: Path,
    dataset_name: str,
    experiment_seed: int,
    split_seed: int,
) -> Path:

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    name = (
        f"{experiment_seed}-"
        f"{split_seed}-"
        f"{timestamp}"
    )

    return (
        results_root
        / dataset_name
        / name
    )


def main() -> None:

    args = _parse_args()

    overrides: dict[
        str,
        Any,
    ] = {}

    if args.mode is not None:
        overrides[
            "experiment.mode"
        ] = args.mode

    if args.experiment_seed is not None:
        overrides[
            "experiment.experiment_seed"
        ] = args.experiment_seed

    if args.split_seed is not None:
        overrides[
            "experiment.split_seed"
        ] = args.split_seed

    config = compose_config(
        dataset_path=(
            args.dataset_config
        ),
        experiment_path=(
            args.experiment_config
        ),
        overrides=overrides,
    )

    run_directory = _run_directory(
        results_root=config.results_root,
        dataset_name=config.dataset_name,
        experiment_seed=(
            config.experiment_seed
        ),
        split_seed=config.split_seed,
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    logging_config = (
        config.section(
            "logging"
        )
    )

    logger = configure_logging(
        run_directory=run_directory,
        level=str(
            logging_config.get(
                "level",
                "INFO",
            )
        ),
        console=bool(
            logging_config.get(
                "console",
                True,
            )
        ),
        save_to_file=bool(
            logging_config.get(
                "save_to_file",
                True,
            )
        ),
    )

    writer = ResultsWriter(
        run_directory
    )

    writer.write_config(
        config.as_dict()
    )

    manifest = {
        "dataset":
            config.dataset_name,

        "timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "git_commit":
            _git_commit(),

        "experiment_seed":
            config.experiment_seed,

        "split_seed":
            config.split_seed,

        "mode":
            config.mode,

        "enabled_methods":
            list(
                config.enabled_methods
            ),

        "dataset_config":
            str(
                Path(
                    args.dataset_config
                ).resolve()
            ),

        "experiment_config":
            str(
                Path(
                    args.experiment_config
                ).resolve()
            ),

        "tuning_config":
            (
                str(
                    Path(
                        args.tuning_config
                    ).resolve()
                )
                if args.tuning_config
                else None
            ),
    }

    writer.write_manifest(
        manifest
    )

    logger.info(
        "Metadata TTA V3"
    )

    logger.info(
        "dataset=%s mode=%s "
        "experiment_seed=%d split_seed=%d",
        config.dataset_name,
        config.mode,
        config.experiment_seed,
        config.split_seed,
    )

    logger.info(
        "run_directory=%s",
        run_directory,
    )

    tuning_config = None

    if config.mode in {
        "tune",
        "tune_and_test",
    }:

        tuning_config = load_yaml(
            args.tuning_config
        )

    loaded_best_overrides = None

    if args.best_overrides is not None:

        loaded_best_overrides = (
            _load_mapping(
                args.best_overrides
            )
        )

    result = run_experiment(
        config,
        tuning_config=tuning_config,
        loaded_best_overrides=(
            loaded_best_overrides
        ),
        run_directory=run_directory,
    )

    # ========================================================
    # PERSIST EFFECTIVE CONFIGURATION
    # ========================================================
    #
    # run_experiment() may return a configuration different
    # from the initial one:
    #
    # - tune_and_test:
    #     selected model / Aux / TTA hyperparameters have been
    #     applied;
    #
    # - test:
    #     explicitly loaded best-overrides may have been
    #     applied.
    #
    # run_config.yaml must describe the configuration that
    # actually produced the final experiment, not merely the
    # configuration supplied at startup.
    #
    # In tune-only mode this still records the configuration
    # returned by the experiment runner.

    writer.write_config(
        result.config.as_dict()
    )

    logger.info(
        "Effective run configuration written to %s",
        run_directory / "run_config.yaml",
    )

    # tune-only intentionally stops before final checkpoints
    # and final test results.
    if result.evaluation is None:

        logger.info(
            "Tuning completed successfully."
        )

        logger.info(
            "run_directory=%s",
            run_directory,
        )

        return

    if result.final_models is None:
        raise RuntimeError(
            "Final evaluation exists but final "
            "models are missing."
        )

    writer.save_source_model(
        result.final_models.single
    )

    writer.save_aux_head(
        result.final_models.double
    )

    writer.write_id_frozen(
        result.evaluation.id_frozen
    )

    writer.write_frozen_ood(
        result.evaluation.ood_frozen
    )

    for (
        method_name,
        method_result,
    ) in result.evaluation.tta.items():

        writer.write_tta_result(
            method_name,
            method_result,
        )

    writer.write_supervised_reference(
        result.evaluation.supervised_reference
    )

    summary = build_final_summary(
        frozen_records=(
            result.evaluation.ood_frozen
        ),
        tta_results=(
            result.evaluation.tta
        ),
        supervised_reference=(
            result.evaluation.supervised_reference
        ),
    )

    writer.write_global_summary(
        summary
    )

    logger.info(
        "Final experiment completed successfully."
    )

    logger.info(
        "Results written to %s",
        run_directory,
    )


if __name__ == "__main__":
    main()