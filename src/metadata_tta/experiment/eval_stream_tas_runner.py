from __future__ import annotations

import copy
from datetime import datetime
from typing import Any

import numpy as np
import torch
from torch import nn

from metadata_tta.config import ExperimentConfig
from metadata_tta.data import build_data_bundle
from metadata_tta.evaluation import (
    EvaluationRecord,
    evaluate_frozen_year,
    evaluate_tta_stream,
)
from metadata_tta.experiment.runner import (
    ExperimentResult,
    _build_double_model,
    _build_single_model,
    _device,
)
from metadata_tta.protocols import (
    EvalStreamTASData,
    OODStreamYear,
    SourceStreamYear,
    build_eval_stream_tas,
)
from metadata_tta.reproducibility import set_seed
from metadata_tta.results import (
    ResultsWriter,
    build_ood_summary,
)
from metadata_tta.training import (
    TrainingSchedule,
    build_training_schedule,
    initialize_double_from_single,
    train_aux_head_only,
    train_single_head,
)
from metadata_tta.tta import get_method_class


def _method_family(
    method_name: str,
) -> str:
    if method_name in {
        "single_head_frozen",
        "tent",
    }:
        return "single"

    return "double"


def _concat_year_arrays(
    years: list[OODStreamYear],
    field_name: str,
) -> np.ndarray:
    arrays = [
        getattr(
            year,
            field_name,
        )
        for year in years
    ]

    if not arrays:
        raise ValueError(
            "Cannot concatenate an empty year list."
        )

    return np.concatenate(
        arrays,
        axis=0,
    )

def _train_source_models_with_yearly_id(
    protocol_data: EvalStreamTASData,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[
    nn.Module | None,
    nn.Module | None,
    list[EvaluationRecord],
]:

    first_year = (
        protocol_data.source_years[0]
    )

    input_dim = int(
        first_year.X_supervised.shape[1]
    )

    double_needed = (
        config.baseline_enabled(
            "double_head"
        )
        or any(
            method_name != "tent"
            for method_name
            in config.enabled_methods
        )
    )

    single_needed = (
        config.baseline_enabled(
            "single_head"
        )
        or config.method_enabled(
            "tent"
        )
        or double_needed
    )

    single_model = (
        _build_single_model(
            input_dim=input_dim,
            config=config,
        ).to(device)
        if single_needed
        else None
    )

    # The double model is built only AFTER the complete
    # single-head source trajectory has finished.
    double_model: nn.Module | None = None

    # Reset RNG after architecture-specific model construction.
    # Final runs are executed one model family at a time, so this
    # makes source-training randomness independent of model init.
    set_seed(config.seed)

    single_optimizer = None

    id_records: list[
        EvaluationRecord
    ] = []

    # ========================================================
    # SEQUENTIAL SOURCE TRAINING
    # ========================================================

    for year_index, year_data in enumerate(
        protocol_data.source_years
    ):

        initialized_from_previous = (
            year_index > 0
        )

        single_schedule = (
            build_training_schedule(
                config=config,
                initialized_from_previous=(
                    initialized_from_previous
                ),
                model_kind="single_head",
            )
        )

        print()
        print(
            "=" * 80
        )
        print(
            f"Source year {year_data.year} | "
            f"train_n="
            f"{len(year_data.y_main_supervised)} | "
            f"val_n="
            f"{len(year_data.y_main_validation)} | "
            f"id_n="
            f"{len(year_data.y_main_id)}"
        )
        print(
            "=" * 80
        )

        # ====================================================
        # SINGLE HEAD
        # ====================================================

        if single_model is not None:

            result = train_single_head(
                model=single_model,

                X=(
                    year_data.X_supervised
                ),

                y_main=(
                    year_data.y_main_supervised
                ),

                schedule=single_schedule,

                device=device,

                optimizer=single_optimizer,

                X_validation=(
                    year_data.X_validation
                ),

                y_main_validation=(
                    year_data.y_main_validation
                ),

                log_prefix=(
                    f"Source {year_data.year} | "
                    "Single Head"
                ),
            )

            single_model = (
                result.model
            )

            single_optimizer = (
                result.optimizer
            )

            if config.baseline_enabled(
                "single_head"
            ):

                id_records.append(
                    evaluate_frozen_year(
                        model=single_model,
                        X=year_data.X_id,
                        y_main=(
                            year_data.y_main_id
                        ),
                        year=year_data.year,
                        method_name=(
                            "single_head_frozen"
                        ),
                        device=device,
                    )
                )

    # ========================================================
    # AUX-ONLY DOUBLE FROM FINAL SINGLE SOURCE MODEL
    # ========================================================

    if double_needed:

        if single_model is None:
            raise RuntimeError(
                "Aux-only double requires a trained "
                "single-head source model."
            )

        set_seed(config.seed)

        double_model = (
            _build_double_model(
                input_dim=input_dim,
                config=config,
            )
            .to(device)
        )

        double_model = initialize_double_from_single(
            single_model=single_model,
            double_model=double_model,
        )

        # Pool ONLY source supervised/validation data.
        X_aux_train = np.concatenate(
            [
                year_data.X_supervised
                for year_data
                in protocol_data.source_years
            ],
            axis=0,
        )

        y_aux_train = np.concatenate(
            [
                year_data.y_aux_supervised
                for year_data
                in protocol_data.source_years
            ],
            axis=0,
        )

        X_aux_validation = np.concatenate(
            [
                year_data.X_validation
                for year_data
                in protocol_data.source_years
            ],
            axis=0,
        )

        y_aux_validation = np.concatenate(
            [
                year_data.y_aux_validation
                for year_data
                in protocol_data.source_years
            ],
            axis=0,
        )

        aux_schedule = build_training_schedule(
            config=config,
            initialized_from_previous=False,
            model_kind="double_head",
        )

        # If early stopping is disabled, setting patience larger
        # than the total number of epochs prevents early stopping.
        aux_patience = (
            aux_schedule.early_stopping_patience
            if aux_schedule.early_stopping_enabled
            else aux_schedule.epochs + 1
        )

        set_seed(config.seed)

        double_model = train_aux_head_only(
            model=double_model,
            X=X_aux_train,
            y_aux=y_aux_train,
            X_validation=X_aux_validation,
            y_aux_validation=y_aux_validation,
            learning_rate=aux_schedule.learning_rate,
            weight_decay=aux_schedule.weight_decay,
            batch_size=aux_schedule.batch_size,
            epochs=aux_schedule.epochs,
            patience=aux_patience,
            min_delta=(
                aux_schedule.early_stopping_min_delta
            ),
            device=device,
            log_prefix="Source | AUX-ONLY DOUBLE",
        )

    # ========================================================
    # AUX-ONLY MAIN-PATH SANITY CHECK
    # ========================================================

    if (
        single_model is not None
        and double_model is not None
    ):

        single_model.eval()
        double_model.eval()

        # OOD features only: no OOD labels are inspected here.
        sanity_X = torch.as_tensor(
            protocol_data.ood_years[0].X_test[:256],
            dtype=torch.float32,
            device=device,
        )

        with torch.no_grad():

            single_logits = single_model(
                sanity_X
            )

            double_logits = double_model(
                sanity_X
            )

        max_abs_diff = float(
            (
                single_logits
                - double_logits
            )
            .abs()
            .max()
            .item()
        )

        predictions_equal = bool(
            torch.equal(
                single_logits.argmax(dim=1),
                double_logits.argmax(dim=1),
            )
        )

        print(
            "FINAL_AUX_ONLY_MAIN_SANITY | "
            f"max_abs_logit_diff={max_abs_diff:.12e} | "
            f"predictions_equal={predictions_equal}"
        )

        if max_abs_diff > 1.0e-7:
            raise RuntimeError(
                "Aux-only double changed the copied "
                "single-head main path."
            )

        if not predictions_equal:
            raise RuntimeError(
                "Aux-only double predictions differ "
                "from single-head predictions."
            )

    if single_model is not None:
        single_model.eval()

    if double_model is not None:
        double_model.eval()

    return (
        single_model,
        double_model,
        id_records,
    )

def _evaluate_supervised_references(
    source_single_model: nn.Module | None,
    protocol_data: EvalStreamTASData,
    config: ExperimentConfig,
    device: torch.device,
) -> list[EvaluationRecord]:
    """
    Build the supervised TAS reference as ONE continual trajectory
    across the OOD years.

    Example:
        source_2012
            -> supervised train 2013 -> evaluate test 2013
            -> supervised train 2014 -> evaluate test 2014
            -> supervised train 2015 -> evaluate test 2015
            -> ...

    The model is copied from the source model only once.

    When training.temporal.reset_optimizer_each_year == False,
    the optimizer state is also carried from one OOD year to the next.

    train_single_head() / train_double_head() already restore the
    best model state AND the corresponding best optimizer state
    when early stopping is enabled, so the next year starts from
    the best checkpoint of the previous year.
    """

    records: list[EvaluationRecord] = []

    # ========================================================
    # INITIALIZE THE SUPERVISED REFERENCES ONLY ONCE
    # ========================================================

    single_reference = (
        copy.deepcopy(
            source_single_model
        ).to(device)
        if source_single_model is not None
        else None
    )

    # New optimizer at the beginning of the OOD supervised
    # reference trajectory.
    #
    # After the first OOD year, the returned optimizer is reused
    # when reset_optimizer_each_year=False.
    single_optimizer = None

    # ========================================================
    # SINGLE CONTINUAL SUPERVISED TRAJECTORY
    # ========================================================

    for year_data in protocol_data.ood_years:

        print()
        print(
            "=" * 80
        )
        print(
            f"Supervised TAS reference | year {year_data.year}"
        )
        print(
            "=" * 80
        )

        # ====================================================
        # SINGLE HEAD REFERENCE
        # ====================================================

        if single_reference is not None:

            schedule = build_training_schedule(
                config=config,
                initialized_from_previous=True,
                model_kind="single_head",
            )

            print()
            print(
                f"TAS supervised Single Head | "
                f"fine-tuning year {year_data.year} | "
                f"train_n="
                f"{len(year_data.y_main_reference_train)} | "
                f"val_n="
                f"{len(year_data.y_main_reference_validation)}"
            )

            result = train_single_head(
                model=single_reference,

                X=(
                    year_data.X_reference_train
                ),

                y_main=(
                    year_data.y_main_reference_train
                ),

                schedule=schedule,

                device=device,

                optimizer=single_optimizer,

                X_validation=(
                    year_data.X_reference_validation
                ),

                y_main_validation=(
                    year_data.y_main_reference_validation
                ),

                log_prefix=(
                    f"TAS {year_data.year} | "
                    "Single Head"
                ),
            )

            # IMPORTANT:
            # train_single_head() has already restored the best
            # validation checkpoint and the corresponding optimizer.
            single_reference = result.model
            single_optimizer = result.optimizer

            records.append(
                evaluate_frozen_year(
                    model=single_reference,
                    X=year_data.X_test,
                    y_main=year_data.y_main_test,
                    year=year_data.year,
                    method_name=(
                        "single_head_supervised_reference"
                    ),
                    device=device,
                )
            )

    return records

def _build_tas_rows(
    ood_records: list[EvaluationRecord],
    supervised_reference_records: list[EvaluationRecord],
) -> list[dict[str, Any]]:

    accuracy_by_key = {
        (
            record.method,
            record.year,
        ): float(
            record.accuracy
        )
        for record in (
            list(ood_records)
            + list(
                supervised_reference_records
            )
        )
    }

    rows: list[
        dict[str, Any]
    ] = []

    for record in ood_records:

        family = _method_family(
            record.method
        )

        frozen_method = (
            "single_head_frozen"
        )
        supervised_method = (
            "single_head_supervised_reference"
        )

        frozen_accuracy = accuracy_by_key.get(
            (
                frozen_method,
                record.year,
            )
        )
        supervised_accuracy = accuracy_by_key.get(
            (
                supervised_method,
                record.year,
            )
        )

        tas_defined = False
        tas_value: float | str = ""

        if (
            frozen_accuracy is not None
            and supervised_accuracy is not None
        ):
            denominator = (
                supervised_accuracy
                - frozen_accuracy
            )

            if abs(denominator) > 1.0e-12:
                tas_value = (
                    float(record.accuracy)
                    - frozen_accuracy
                ) / denominator
                tas_defined = True

        rows.append(
            {
                "year":
                    int(record.year),

                "method":
                    record.method,

                "family":
                    family,

                "accuracy":
                    float(record.accuracy),

                "frozen_reference_method":
                    frozen_method,

                "frozen_reference_accuracy":
                    (
                        ""
                        if frozen_accuracy is None
                        else float(frozen_accuracy)
                    ),

                "supervised_reference_method":
                    supervised_method,

                "supervised_reference_accuracy":
                    (
                        ""
                        if supervised_accuracy is None
                        else float(supervised_accuracy)
                    ),

                "tas":
                    tas_value,

                "tas_defined":
                    bool(tas_defined),
            }
        )

    return rows


def run_eval_stream_tas(
    config: ExperimentConfig,
) -> ExperimentResult:

    set_seed(
        config.seed
    )

    device = _device()

    print(
        f"Device: {device}"
    )
    print(
        "Protocol: eval_stream_tas"
    )

    bundle = build_data_bundle(
        config
    )

    protocol_data = build_eval_stream_tas(
        bundle=bundle,
        config=config,
    )

    (
        single_model,
        double_model,
        id_records,
    ) = _train_source_models_with_yearly_id(
        protocol_data=protocol_data,
        config=config,
        device=device,
    )

    if (
        config.method_enabled(
            "tent"
        )
        and single_model is None
    ):
        raise RuntimeError(
            "TENT requires a SingleHead source model."
        )

    double_required = (
        config.baseline_enabled(
            "double_head"
        )
        or any(
            method_name != "tent"
            for method_name
            in config.enabled_methods
        )
    )

    if (
        double_required
        and double_model is None
    ):
        raise RuntimeError(
            "DoubleHead source model is required "
            "for double-head baselines and metadata TTA."
        )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_directory = (
        config.results_root
        / config.dataset_name
        / config.protocol_name
        / (
            f"{config.experiment_name}_"
            f"{timestamp}"
        )
    )

    writer = ResultsWriter(
        run_directory
    )

    writer.write_config(
        config.as_dict()
    )

    metadata_pre_prediction_episodic = bool(
        config.method_enabled("metadata")
        and config.section("metadata").get(
            "episodic_pre_prediction",
            False,
        )
    )

    temporal_gradient_pre_prediction = bool(
        config.method_enabled(
            "temporal_gradient"
        )
    )

    writer.write_manifest(
        {
            "experiment":
                config.experiment_name,

            "dataset":
                config.dataset_name,

            "protocol":
                config.protocol_name,

            "device":
                str(device),

            "seed":
                config.seed,

            "source_id_policy":
                (
                    "evaluate each source-year held-out "
                    "split immediately after training that year"
                ),

            "ood_policy":
                (
                    "per-year 80/20 split; 20% test stream; "
                    "80% supervised TAS reference pool"
                ),

            "tta_order":
                (
                    "method_specific"
                    if (
                        metadata_pre_prediction_episodic
                        and temporal_gradient_pre_prediction
                    )
                    else (
                        "metadata_update_then_predict"
                        if (
                            metadata_pre_prediction_episodic
                            or temporal_gradient_pre_prediction
                        )
                        else "predict_then_update"
                    )
                ),

            "tta_state":
                (
                    "method_specific"
                    if (
                        metadata_pre_prediction_episodic
                        and temporal_gradient_pre_prediction
                    )
                    else (
                        "reset_to_source_each_sample"
                        if metadata_pre_prediction_episodic
                        else "carries_across_ood_years"
                    )
                ),

            "tta_order_by_method": {
                "metadata": (
                    "update_then_predict"
                    if metadata_pre_prediction_episodic
                    else "predict_then_update"
                ),
                "temporal_gradient":
                    "update_then_predict",
            },

            "tta_state_by_method": {
                "metadata": (
                    "reset_to_source_each_sample"
                    if metadata_pre_prediction_episodic
                    else "carries_across_ood_years"
                ),
                "temporal_gradient":
                    "carries_across_ood_years",
            },

            "supervised_reference_policy":
                "single_continual_trajectory_across_ood_years",

            "supervised_reference_state":
                "model_and_optimizer_carry_across_ood_years",

            "tent_source_model":
                "single_head",

            "metadata_tta_source_model":
                "double_head",

            "pca_active":
                False,
        }
    )

    ood_records: list[
        EvaluationRecord
    ] = []

    # ========================================================
    # OOD FROZEN BASELINES
    # ========================================================

    for year_data in protocol_data.ood_years:

        print()
        print(
            f"OOD frozen year {year_data.year} | "
            f"test_n={len(year_data.y_main_test)}"
        )

        if (
            single_model is not None
            and config.baseline_enabled(
                "single_head"
            )
        ):
            ood_records.append(
                evaluate_frozen_year(
                    model=single_model,
                    X=year_data.X_test,
                    y_main=(
                        year_data.y_main_test
                    ),
                    year=year_data.year,
                    method_name=(
                        "single_head_frozen"
                    ),
                    device=device,
                )
            )

        if (
            double_model is not None
            and config.baseline_enabled(
                "double_head"
            )
        ):
            ood_records.append(
                evaluate_frozen_year(
                    model=double_model,
                    X=year_data.X_test,
                    y_main=(
                        year_data.y_main_test
                    ),
                    year=year_data.year,
                    method_name=(
                        "double_head_frozen"
                    ),
                    device=device,
                )
            )

    # ========================================================
    # TTA METHODS
    # ========================================================

    stream = [
        (
            year_data.year,
            year_data.X_test,
            year_data.y_main_test,
            year_data.y_aux_test,
        )
        for year_data
        in protocol_data.ood_years
    ]

    for method_name in config.enabled_methods:

        method_class = get_method_class(
            method_name
        )

        method_config = copy.deepcopy(
            config.section(
                method_name
            )
        )

        if method_name == "tent":

            if single_model is None:
                raise RuntimeError(
                    "TENT requires a SingleHead "
                    "source model."
                )

            source_model = single_model

        else:

            if double_model is None:
                raise RuntimeError(
                    f"{method_name} requires a "
                    "DoubleHead source model."
                )

            source_model = double_model

        print()
        print(
            f"TTA method {method_name} | "
            "state carries across OOD years"
        )

        method = method_class(
            source_model=source_model,
            config=method_config,
            device=device,
        )

        result = evaluate_tta_stream(
            method=method,
            years=stream,
            reset_each_year=False,
        )

        ood_records.extend(
            result.records
        )

        if config.section(
            "output"
        ).get(
            "save_diagnostics",
            True,
        ):
            writer.write_diagnostics(
                method_name=method_name,
                diagnostics=result.diagnostics,
            )

        # ========================================================
        # SUPERVISED TAS REFERENCES
        # ========================================================

        compute_supervised_upper_bound = bool(
            config.section("output").get(
                "compute_supervised_upper_bound",
                True,
            )
        )

        if compute_supervised_upper_bound:
            supervised_reference_records = (
                _evaluate_supervised_references(
                    source_single_model=single_model,
                    protocol_data=protocol_data,
                    config=config,
                    device=device,
                )
            )
        else:
            supervised_reference_records = []

            print()
            print(
                "Supervised TAS reference skipped "
                "(output.compute_supervised_upper_bound=false)."
            )

    tas_rows = _build_tas_rows(
        ood_records=ood_records,
        supervised_reference_records=(
            supervised_reference_records
        ),
    )

    # ========================================================
    # WRITE RESULTS
    # ========================================================

    writer.write_records(
        "id.csv",
        id_records,
    )

    writer.write_records(
        "ood.csv",
        ood_records,
    )

    if compute_supervised_upper_bound:
        writer.write_records(
            "supervised_reference.csv",
            supervised_reference_records,
        )

    writer.write_tas_rows(
        tas_rows
    )

    reference_method = "single_head_frozen"

    summary = build_ood_summary(
        records=ood_records,
        reference_method=reference_method,
    )

    writer.write_summary(
        summary
    )

    print(
        f"Results: {run_directory}"
    )

    return ExperimentResult(
        run_directory=run_directory,
        id_records=tuple(
            id_records
        ),
        ood_records=tuple(
            ood_records
        ),
    )