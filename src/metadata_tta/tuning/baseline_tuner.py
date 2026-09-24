from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping

import numpy as np
import torch

from metadata_tta.config import (
    ExperimentConfig,
)
from metadata_tta.data import (
    DataBundle,
    build_data_bundle,
    build_split_indices,
)
from metadata_tta.evaluation import (
    evaluate_frozen_year,
)
from metadata_tta.experiment.runner import (
    _device,
)
from metadata_tta.reproducibility import (
    set_seed,
)
from metadata_tta.training import (
    build_training_schedule,
    create_double_head_model,
    create_single_head_model,
    get_aux_loss_weight,
    train_double_head,
    train_single_head,
)

from .config import (
    apply_overrides,
)
from .inheritance import (
    inherited_overrides_from_best,
    merge_overrides,
)
from .results import (
    ensure_directory,
    save_best_result,
    save_trial_rows,
)
from .search import (
    build_search_strategy,
)


@dataclass(frozen=True)
class BaselineTuningResult:
    baseline_name: str
    best_score: float
    best_parameters: dict[str, Any]
    best_overrides: dict[str, Any]
    output_directory: Path


# ============================================================
# YEAR RANGE
# ============================================================


def _source_year_range(
    config: ExperimentConfig,
    section: Mapping[str, Any],
) -> list[int]:

    protocol = config.section(
        "protocol"
    )

    start = int(
        section.get(
            "start_year",
            protocol[
                "source_start_year"
            ],
        )
    )

    end = int(
        section.get(
            "end_year",
            protocol[
                "source_end_year"
            ],
        )
    )

    if start > end:
        raise ValueError(
            "Baseline tuning start_year "
            "must be <= end_year."
        )

    return list(
        range(
            start,
            end + 1,
        )
    )


# ============================================================
# YEARLY TRAIN / VALIDATION SPLIT
# ============================================================


def _year_split(
    bundle: DataBundle,
    config: ExperimentConfig,
    year: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    data = bundle.year(
        year
    )

    split_config = config.get(
        "dataset",
        "split",
    )

    split = build_split_indices(
        y_main=data.y_main,
        test_size=float(
            split_config[
                "test_size"
            ]
        ),
        validation_size_within_train=float(
            split_config[
                "validation_size_within_train"
            ]
        ),
        seed=int(
            split_config[
                "seed"
            ]
        ),
    )

    if len(
        split.validation
    ) == 0:
        raise ValueError(
            "Baseline tuning requires a "
            "non-empty yearly validation set."
        )

    return (
        data.X[
            split.train
        ],
        data.y_main[
            split.train
        ],
        data.y_aux[
            split.train
        ],
        data.X[
            split.validation
        ],
        data.y_main[
            split.validation
        ],
        data.y_aux[
            split.validation
        ],
    )


# ============================================================
# SINGLE-HEAD TRIAL
# ============================================================

def _run_single_trial(
    config: ExperimentConfig,
    bundle: DataBundle,
    years: list[int],
    device: torch.device,
) -> dict[str, float]:

    model = (
        create_single_head_model(
            input_dim=bundle.feature_dim,
            config=config,
        )
        .to(device)
    )

    # Reset RNG after model construction so training randomness
    # is independent of architecture-specific parameter initialization.
    set_seed(config.seed)

    optimizer = None

    accuracies: dict[
        str,
        float,
    ] = {}

    for index, year in enumerate(
        years
    ):

        (
            X_train,
            y_main_train,
            _,
            X_validation,
            y_main_validation,
            _,
        ) = _year_split(
            bundle=bundle,
            config=config,
            year=year,
        )

        schedule = (
            build_training_schedule(
                config=config,
                initialized_from_previous=(
                    index > 0
                ),
                model_kind="single_head",
            )
        )

        result = train_single_head(
            model=model,

            X=X_train,

            y_main=y_main_train,

            schedule=schedule,

            device=device,

            optimizer=optimizer,

            X_validation=(
                X_validation
            ),

            y_main_validation=(
                y_main_validation
            ),

            log_prefix=(
                f"Baseline tuning {year} | "
                "Single Head"
            ),
        )

        model = (
            result.model
        )

        optimizer = (
            result.optimizer
        )

        record = (
            evaluate_frozen_year(
                model=model,
                X=X_validation,
                y_main=(
                    y_main_validation
                ),
                year=year,
                method_name=(
                    "single_head_validation"
                ),
                device=device,
            )
        )

        accuracies[
            str(
                year
            )
        ] = float(
            record.accuracy
        )

    return accuracies

# ============================================================
# DOUBLE-HEAD TRIAL
# ============================================================

def _run_double_trial(
    config: ExperimentConfig,
    bundle: DataBundle,
    years: list[int],
    device: torch.device,
) -> dict[str, float]:

    model = (
        create_double_head_model(
            input_dim=bundle.feature_dim,
            config=config,
        )
        .to(device)
    )

    # Reset RNG after model construction so training randomness
    # matches the single-head baseline.
    set_seed(config.seed)

    optimizer = None

    aux_loss_weight = (
        get_aux_loss_weight(
            config
        )
    )

    accuracies: dict[
        str,
        float,
    ] = {}

    for index, year in enumerate(
        years
    ):

        (
            X_train,
            y_main_train,
            y_aux_train,
            X_validation,
            y_main_validation,
            y_aux_validation,
        ) = _year_split(
            bundle=bundle,
            config=config,
            year=year,
        )

        schedule = (
            build_training_schedule(
                config=config,
                initialized_from_previous=(
                    index > 0
                ),
                model_kind="double_head",
            )
        )

        result = train_double_head(
            model=model,

            X=X_train,

            y_main=(
                y_main_train
            ),

            y_aux=(
                y_aux_train
            ),

            schedule=schedule,

            aux_loss_weight=(
                aux_loss_weight
            ),

            device=device,

            optimizer=optimizer,

            X_validation=(
                X_validation
            ),

            y_main_validation=(
                y_main_validation
            ),

            y_aux_validation=(
                y_aux_validation
            ),

            log_prefix=(
                f"Baseline tuning {year} | "
                "Double Head"
            ),
        )

        model = (
            result.model
        )

        optimizer = (
            result.optimizer
        )

        record = (
            evaluate_frozen_year(
                model=model,
                X=X_validation,
                y_main=(
                    y_main_validation
                ),
                year=year,
                method_name=(
                    "double_head_validation"
                ),
                device=device,
            )
        )

        # ----------------------------------------------------
        # Auxiliary-head diagnostic.
        # Measure whether metadata is actually learnable.
        # This does NOT affect training or model selection.
        # ----------------------------------------------------

        was_training = model.training
        model.eval()

        def _aux_metrics(X_eval, y_aux_eval):
            X_tensor = torch.as_tensor(
                X_eval,
                dtype=torch.float32,
                device=device,
            )
            y_tensor = torch.as_tensor(
                y_aux_eval,
                dtype=torch.long,
                device=device,
            )

            with torch.no_grad():
                _, aux_logits = model.forward_both(X_tensor)

                aux_loss = torch.nn.functional.cross_entropy(
                    aux_logits,
                    y_tensor,
                )

                aux_predictions = aux_logits.argmax(dim=1)

                aux_accuracy = (
                    aux_predictions == y_tensor
                ).float().mean()

                counts = torch.bincount(y_tensor)
                majority_accuracy = (
                    counts.max().float()
                    / max(len(y_tensor), 1)
                )

            return (
                float(aux_loss.item()),
                float(aux_accuracy.item()),
                float(majority_accuracy.item()),
            )

        (
            train_aux_loss,
            train_aux_accuracy,
            train_aux_majority,
        ) = _aux_metrics(
            X_train,
            y_aux_train,
        )

        (
            val_aux_loss,
            val_aux_accuracy,
            val_aux_majority,
        ) = _aux_metrics(
            X_validation,
            y_aux_validation,
        )

        if was_training:
            model.train()

        print(
            "AUX_DIAG | "
            f"year={year} | "
            f"aux_weight={aux_loss_weight:.6g} | "
            f"train_acc={train_aux_accuracy:.6f} | "
            f"train_loss={train_aux_loss:.6f} | "
            f"train_majority={train_aux_majority:.6f} | "
            f"val_acc={val_aux_accuracy:.6f} | "
            f"val_loss={val_aux_loss:.6f} | "
            f"val_majority={val_aux_majority:.6f}"
        )

        accuracies[
            str(
                year
            )
        ] = float(
            record.accuracy
        )

    return accuracies

# ============================================================
# INHERITANCE
# ============================================================


def _baseline_parent_overrides(
    *,
    baseline_name: str,
    tuning_section: Mapping[str, Any],
    output_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:

    parent_name = tuning_section.get(
        "inherits_from"
    )

    if parent_name is None:
        return (
            {},
            {},
        )

    parent_name = str(
        parent_name
    )

    mapping = tuning_section.get(
        "inherit_parameters",
        {},
    )

    if not isinstance(
        mapping,
        Mapping,
    ):
        raise ValueError(
            f"{baseline_name}.inherit_parameters "
            "must be a mapping."
        )

    parent_best_path = (
        output_root
        / "baselines"
        / parent_name
        / "best.yaml"
    )

    if not parent_best_path.is_file():
        raise FileNotFoundError(
            f"{baseline_name} inherits from "
            f"{parent_name}, but parent result "
            f"does not exist: {parent_best_path}"
        )

    inherited = (
        inherited_overrides_from_best(
            best_path=parent_best_path,
            parameter_mapping=mapping,
        )
    )

    return (
        inherited,
        {
            "inherits_from":
                parent_name,
            "parent_best_path":
                str(
                    parent_best_path
                ),
        },
    )


# ============================================================
# TUNE ONE BASELINE
# ============================================================


def tune_baseline(
    *,
    baseline_name: str,
    base_config: ExperimentConfig,
    tuning_section: Mapping[str, Any],
    bundle: DataBundle,
    output_root: Path,
) -> BaselineTuningResult:

    if baseline_name not in {
        "single_head",
        "double_head",
    }:
        raise ValueError(
            f"Unknown baseline: "
            f"{baseline_name}"
        )

    search_name = str(
        tuning_section.get(
            "search_strategy",
            "grid",
        )
    )

    strategy = (
        build_search_strategy(
            search_name
        )
    )

    space = tuning_section.get(
        "space",
        {},
    )

    if not isinstance(
        space,
        Mapping,
    ):
        raise ValueError(
            f"{baseline_name}.space "
            "must be a mapping."
        )

    years = _source_year_range(
        config=base_config,
        section=tuning_section,
    )

    output_directory = (
        ensure_directory(
            output_root
            / "baselines"
            / baseline_name
        )
    )

    (
        inherited_overrides,
        inheritance_info,
    ) = _baseline_parent_overrides(
        baseline_name=baseline_name,
        tuning_section=tuning_section,
        output_root=output_root,
    )

    if inherited_overrides:
        print()
        print(
            f"{baseline_name} inherited overrides:"
        )

        for path, value in (
            inherited_overrides.items()
        ):
            print(
                f"  {path} = {value}"
            )

    device = _device()

    trial_rows: list[
        dict[str, Any]
    ] = []

    best_score = float(
        "-inf"
    )

    best_parameters: dict[
        str,
        Any,
    ] = {}

    best_overrides: dict[
        str,
        Any,
    ] = {}

    for trial in strategy.generate(
        space
    ):

        print()
        print(
            "=" * 80
        )
        print(
            f"BASELINE TUNING | "
            f"{baseline_name} | "
            f"trial={trial.trial_id}"
        )
        print(
            trial.parameters
        )
        print(
            "=" * 80
        )

        # ----------------------------------------------------
        # Parent configuration first, current trial second.
        #
        # Current trial wins if the same path appears twice.
        # This also allows future local refinement.
        # ----------------------------------------------------

        complete_overrides = (
            merge_overrides(
                inherited_overrides,
                trial.overrides,
            )
        )

        trial_config = (
            apply_overrides(
                config=base_config,
                overrides=(
                    complete_overrides
                ),
            )
        )

        # Every trial must start from the exact same RNG state.
        set_seed(
            trial_config.seed
        )

        if baseline_name == (
            "single_head"
        ):

            yearly = (
                _run_single_trial(
                    config=trial_config,
                    bundle=bundle,
                    years=years,
                    device=device,
                )
            )

        else:

            yearly = (
                _run_double_trial(
                    config=trial_config,
                    bundle=bundle,
                    years=years,
                    device=device,
                )
            )

        values = list(
            yearly.values()
        )

        score = float(
            mean(values)
        )

        std = float(
            pstdev(values)
            if len(values) > 1
            else 0.0
        )

        row: dict[
            str,
            Any,
        ] = {
            "trial_id":
                trial.trial_id,
            "mean_validation_accuracy":
                score,
            "std_validation_accuracy":
                std,
        }

        for key, value in (
            trial.parameters.items()
        ):
            row[
                f"param_{key}"
            ] = value

        for path, value in (
            inherited_overrides.items()
        ):
            row[
                f"inherited_{path}"
            ] = value

        for year, accuracy in (
            yearly.items()
        ):
            row[
                f"val_accuracy_{year}"
            ] = accuracy

        trial_rows.append(
            row
        )

        if score > best_score:
            best_score = score

            # Keep only parameters actually searched in this
            # stage as the human-readable parameter result.
            best_parameters = dict(
                trial.parameters
            )

            # But save the COMPLETE effective override set,
            # including inherited parameters.
            best_overrides = dict(
                complete_overrides
            )

        save_trial_rows(
            output_directory
            / "trials.csv",
            trial_rows,
        )

    save_best_result(
        output_directory
        / "best.yaml",
        kind="baseline",
        name=baseline_name,
        score_name=(
            "mean_validation_accuracy"
        ),
        score=best_score,
        parameters=best_parameters,
        overrides=best_overrides,
        extra={
            "years":
                years,
            **inheritance_info,
        },
    )

    return BaselineTuningResult(
        baseline_name=(
            baseline_name
        ),
        best_score=best_score,
        best_parameters=(
            best_parameters
        ),
        best_overrides=(
            best_overrides
        ),
        output_directory=(
            output_directory
        ),
    )


# ============================================================
# STAGED BASELINE PIPELINE
# ============================================================


def tune_baselines(
    *,
    base_config: ExperimentConfig,
    tuning_config: Mapping[str, Any],
    output_root: Path,
) -> dict[
    str,
    BaselineTuningResult,
]:

    baseline_config = (
        tuning_config.get(
            "baseline",
            {},
        )
    )

    if not isinstance(
        baseline_config,
        Mapping,
    ):
        raise ValueError(
            "baseline tuning section "
            "must be a mapping."
        )

    # Dataset is loaded once.
    bundle = build_data_bundle(
        base_config
    )

    results: dict[
        str,
        BaselineTuningResult,
    ] = {}

    # --------------------------------------------------------
    # Explicit dependency order.
    #
    # DO NOT rely on YAML dictionary order for experiment
    # semantics.
    # --------------------------------------------------------

    ordered_stages = (
        "single_head",
        "double_head",
    )

    for baseline_name in (
        ordered_stages
    ):

        section = baseline_config.get(
            baseline_name
        )

        if section is None:
            continue

        if not isinstance(
            section,
            Mapping,
        ):
            raise ValueError(
                f"baseline.{baseline_name} "
                "must be a mapping."
            )

        if not bool(
            section.get(
                "enabled",
                True,
            )
        ):
            continue

        results[
            baseline_name
        ] = tune_baseline(
            baseline_name=(
                baseline_name
            ),
            base_config=(
                base_config
            ),
            tuning_section=(
                section
            ),
            bundle=bundle,
            output_root=(
                output_root
            ),
        )

    return results