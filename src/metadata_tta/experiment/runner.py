from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from metadata_tta.config import ExperimentConfig
from metadata_tta.data import (
    DataBundle,
    build_data_bundle,
)
from metadata_tta.evaluation import (
    EvaluationRecord,
    EvaluationResult,
    evaluate_frozen_year,
    evaluate_tta_stream,
)
from metadata_tta.protocols import (
    SplitName,
    V3Protocol,
)
from metadata_tta.reproducibility import set_seed
from metadata_tta.training import (
    build_training_schedule,
    create_double_head_model,
    create_single_head_model,
    initialize_double_from_single,
    train_aux_head_only,
    train_single_head,
)
from metadata_tta.tta import (
    get_method_class,
    method_is_registered,
)


TTA_METHOD_ORDER = (
    "metadata_episodic",
    "metadata_cumulative",
    "metadata_cumulative_annual_reset",
    "metadata_cumulative_drift_reset",
    "metadata_cumulative_drift_annual_reset",
    "temporal_gradient_ema",
    "tent",
)


@dataclass(frozen=True)
class FinalModels:
    single: nn.Module
    double: nn.Module


@dataclass(frozen=True)
class FinalEvaluation:
    id_frozen: tuple[EvaluationRecord, ...]
    ood_frozen: tuple[EvaluationRecord, ...]
    tta: dict[str, EvaluationResult]
    supervised_reference: tuple[EvaluationRecord, ...]


@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    run_directory: Path | None
    final_models: FinalModels | None
    evaluation: FinalEvaluation | None
    tuning_result: Any | None = None


def _device() -> torch.device:

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def _build_protocol(
    *,
    bundle: DataBundle,
    config: ExperimentConfig,
) -> V3Protocol:

    protocol = V3Protocol(
        bundle=bundle,
        dataset_config=config.section("dataset"),
        protocol_config=config.section("protocol"),
        split_seed=config.split_seed,
    )

    if bool(
        config.get(
            "checks",
            "verify_split_disjointness",
            default=True,
        )
    ):
        protocol.verify_split_disjointness()

    return protocol


def _train_temporal_single(
    *,
    config: ExperimentConfig,
    bundle: DataBundle,
    supervised_slices: tuple,
    device: torch.device,
    log_prefix: str,
) -> nn.Module:
    """
    Train a fresh Single Head sequentially across source years.

    First source year:
        base / single-head schedule.

    Later source years:
        temporal schedule.

    Train = 70%
    Validation = 10%
    Test is never passed here.
    """

    if not supervised_slices:
        raise RuntimeError(
            "Cannot train source model on an empty source."
        )

    set_seed(
        config.experiment_seed
    )

    model = (
        create_single_head_model(
            input_dim=bundle.feature_dim,
            config=config,
        )
        .to(device)
    )

    optimizer = None

    for index, data in enumerate(
        supervised_slices
    ):

        schedule = build_training_schedule(
            config=config,
            initialized_from_previous=(
                index > 0
            ),
            model_kind="single_head",
        )

        print()
        print(
            f"[{log_prefix}] "
            f"source year {data.year} | "
            f"train={len(data.X_train)} | "
            f"val={len(data.X_validation)}"
        )

        result = train_single_head(
            model=model,
            X=data.X_train,
            y_main=data.y_main_train,
            X_validation=data.X_validation,
            y_main_validation=(
                data.y_main_validation
            ),
            schedule=schedule,
            device=device,
            optimizer=optimizer,
            log_prefix=(
                f"{log_prefix}:{data.year}"
            ),
        )

        model = result.model
        optimizer = result.optimizer

    model.eval()

    return model


def _assert_single_double_equivalence(
    *,
    single_model: nn.Module,
    double_model: nn.Module,
    X: np.ndarray,
    device: torch.device,
) -> None:

    if len(X) == 0:
        raise RuntimeError(
            "Cannot check Single/Double equivalence "
            "on empty data."
        )

    n = min(
        64,
        len(X),
    )

    x = torch.as_tensor(
        X[:n],
        dtype=torch.float32,
        device=device,
    )

    single_model.eval()
    double_model.eval()

    with torch.no_grad():

        single_logits = single_model(
            x
        )

        double_logits = double_model(
            x
        )

    if not torch.allclose(
        single_logits,
        double_logits,
        rtol=1.0e-5,
        atol=1.0e-6,
    ):
        max_difference = float(
            (
                single_logits
                - double_logits
            )
            .abs()
            .max()
            .item()
        )

        raise RuntimeError(
            "Single/Double main predictions are not "
            "identical immediately after initialization. "
            f"max_abs_difference={max_difference:.8e}"
        )

    print(
        "[CHECK][PASS] Single/Double main-path "
        "equivalence before Aux training"
    )


def _concatenate_aux_source(
    supervised_slices: tuple,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    if not supervised_slices:
        raise RuntimeError(
            "Cannot build Aux Head data from empty source."
        )

    X_train = np.concatenate(
        [
            item.X_train
            for item in supervised_slices
        ],
        axis=0,
    )

    y_aux_train = np.concatenate(
        [
            item.y_aux_train
            for item in supervised_slices
        ],
        axis=0,
    )

    X_validation = np.concatenate(
        [
            item.X_validation
            for item in supervised_slices
        ],
        axis=0,
    )

    y_aux_validation = np.concatenate(
        [
            item.y_aux_validation
            for item in supervised_slices
        ],
        axis=0,
    )

    return (
        X_train,
        y_aux_train,
        X_validation,
        y_aux_validation,
    )


def _train_final_aux_head(
    *,
    config: ExperimentConfig,
    bundle: DataBundle,
    source_single: nn.Module,
    supervised_slices: tuple,
    device: torch.device,
) -> nn.Module:

    double_model = (
        create_double_head_model(
            input_dim=bundle.feature_dim,
            config=config,
        )
        .to(device)
    )

    double_model = initialize_double_from_single(
        single_model=source_single,
        double_model=double_model,
    )

    first_source = supervised_slices[0]

    _assert_single_double_equivalence(
        single_model=source_single,
        double_model=double_model,
        X=first_source.X_validation,
        device=device,
    )

    (
        X_train,
        y_aux_train,
        X_validation,
        y_aux_validation,
    ) = _concatenate_aux_source(
        supervised_slices
    )

    training = config.section(
        "training"
    )

    aux_config = dict(
        training["double_head"]
    )

    base_config = dict(
        training["base"]
    )

    early_stopping = dict(
        aux_config.get(
            "early_stopping",
            base_config.get(
                "early_stopping",
                {},
            ),
        )
    )

    print()
    print(
        "[FINAL:AUX] "
        f"train={len(X_train)} | "
        f"val={len(X_validation)}"
    )

    double_model = train_aux_head_only(
        model=double_model,
        X=X_train,
        y_aux=y_aux_train,
        X_validation=X_validation,
        y_aux_validation=y_aux_validation,
        learning_rate=float(
            aux_config.get(
                "learning_rate",
                base_config["learning_rate"],
            )
        ),
        weight_decay=float(
            aux_config.get(
                "weight_decay",
                base_config["weight_decay"],
            )
        ),
        batch_size=int(
            aux_config.get(
                "batch_size",
                base_config["batch_size"],
            )
        ),
        epochs=int(
            aux_config.get(
                "epochs",
                base_config["epochs"],
            )
        ),
        patience=int(
            early_stopping.get(
                "patience",
                5,
            )
        ),
        min_delta=float(
            early_stopping.get(
                "min_delta",
                1.0e-4,
            )
        ),
        device=device,
        log_prefix="FINAL:AUX",
    )

    return double_model


def _train_final_models(
    *,
    config: ExperimentConfig,
    bundle: DataBundle,
    protocol: V3Protocol,
    device: torch.device,
) -> FinalModels:
    """
    IMPORTANT:
    This function always constructs a NEW source model.

    It receives no model from tuning. Therefore tune_and_test
    cannot accidentally reuse a pseudo-source tuning checkpoint.
    """

    print()
    print("=" * 80)
    print("FINAL RETRAINING FROM SCRATCH")
    print("=" * 80)

    set_seed(
        config.experiment_seed
    )

    source_slices = (
        protocol.final_source_supervised()
    )

    single = _train_temporal_single(
        config=config,
        bundle=bundle,
        supervised_slices=source_slices,
        device=device,
        log_prefix="FINAL:SINGLE",
    )

    double = _train_final_aux_head(
        config=config,
        bundle=bundle,
        source_single=single,
        supervised_slices=source_slices,
        device=device,
    )

    single.eval()
    double.eval()

    return FinalModels(
        single=single,
        double=double,
    )


def _evaluate_id_frozen(
    *,
    model: nn.Module,
    protocol: V3Protocol,
    device: torch.device,
) -> tuple[EvaluationRecord, ...]:

    records = []

    for year in (
        protocol.years.final_source
    ):

        stream = protocol.stream_slice(
            year=year,
            split_name=SplitName.TEST,
        )

        record = evaluate_frozen_year(
            model=model,
            X=stream.X,
            y_main=stream.y_main,
            year=year,
            method_name="frozen",
            device=device,
        )

        records.append(
            record
        )

        print(
            f"[ID][frozen] year={year} | "
            f"accuracy={record.accuracy:.4f} | "
            f"n={record.n_samples}"
        )

    return tuple(
        records
    )


def _evaluate_ood_frozen(
    *,
    model: nn.Module,
    protocol: V3Protocol,
    device: torch.device,
) -> tuple[EvaluationRecord, ...]:

    records = []

    for stream in (
        protocol.real_ood_test_stream()
    ):

        record = evaluate_frozen_year(
            model=model,
            X=stream.X,
            y_main=stream.y_main,
            year=stream.year,
            method_name="frozen",
            device=device,
        )

        records.append(
            record
        )

        print(
            f"[OOD][frozen] year={stream.year} | "
            f"accuracy={record.accuracy:.4f} | "
            f"n={record.n_samples}"
        )

    return tuple(
        records
    )


def _method_source_model(
    *,
    method_name: str,
    models: FinalModels,
) -> nn.Module:

    if method_name == "tent":
        return models.single

    return models.double


def _evaluate_tta_methods(
    *,
    config: ExperimentConfig,
    models: FinalModels,
    protocol: V3Protocol,
    device: torch.device,
) -> dict[
    str,
    EvaluationResult,
]:

    stream = protocol.evaluator_stream(
        protocol.real_ood_test_stream()
    )

    results = {}

    for method_name in TTA_METHOD_ORDER:

        if not config.method_enabled(
            method_name
        ):
            continue

        if not method_is_registered(
            method_name
        ):
            raise RuntimeError(
                f"Enabled method {method_name!r} "
                "is not registered."
            )

        source_model = _method_source_model(
            method_name=method_name,
            models=models,
        )

        method_class = get_method_class(
            method_name
        )

        method = method_class(
            source_model=source_model,
            config=config.method_config(
                method_name
            ),
            device=device,
        )

        print()
        print(
            f"[OOD][TTA] {method_name}"
        )

        result = evaluate_tta_stream(
            method=method,
            years=stream,
        )

        results[
            method_name
        ] = result

        for record in result.records:

            print(
                f"[OOD][{method_name}] "
                f"year={record.year} | "
                f"accuracy={record.accuracy:.4f} | "
                f"n={record.n_samples}"
            )

    return results


def _supervised_reference_schedule(
    *,
    config: ExperimentConfig,
):
    """
    Supervised OOD reference starts from the final source model,
    therefore it uses the temporal/fine-tuning schedule.

    Each OOD year receives a fresh optimizer because each year
    is an independent oracle experiment.
    """

    return build_training_schedule(
        config=config,
        initialized_from_previous=True,
        model_kind="single_head",
    )


def _evaluate_supervised_reference(
    *,
    config: ExperimentConfig,
    source_model: nn.Module,
    protocol: V3Protocol,
    device: torch.device,
) -> tuple[EvaluationRecord, ...]:
    """
    Option A, fixed protocol:

    For every OOD year independently:

        deepcopy SAME final source model
        train on that year's 70% main labels
        early-stop on that year's 10%
        evaluate on that year's 20%

    No supervised state propagates between OOD years.
    """

    if not config.method_enabled(
        "supervised_reference"
    ):
        return tuple()

    reference_slices = {
        item.year: item
        for item
        in protocol.real_ood_supervised_reference()
    }

    test_slices = {
        item.year: item
        for item
        in protocol.real_ood_test_stream()
    }

    if (
        set(reference_slices)
        != set(test_slices)
    ):
        raise RuntimeError(
            "Supervised-reference train/validation years "
            "do not match OOD test years."
        )

    records = []

    pristine_source_state = copy.deepcopy(
        source_model.state_dict()
    )

    for year in (
        protocol.years.real_ood
    ):

        train_data = reference_slices[
            year
        ]

        test_data = test_slices[
            year
        ]

        # Independent deepcopy of exactly the same final source.
        model = copy.deepcopy(
            source_model
        ).to(device)

        # Fail hard if a future edit accidentally passes a
        # previously fine-tuned model into this loop.
        current_state = model.state_dict()

        for key, expected in (
            pristine_source_state.items()
        ):

            actual = current_state[
                key
            ]

            if not torch.equal(
                actual.detach().cpu(),
                expected.detach().cpu(),
            ):
                raise RuntimeError(
                    "Supervised reference did not start "
                    "from the pristine final source model."
                )

        schedule = (
            _supervised_reference_schedule(
                config=config
            )
        )

        print()
        print(
            f"[SUPERVISED_REFERENCE] "
            f"year={year} | "
            f"train={len(train_data.X_train)} | "
            f"val={len(train_data.X_validation)} | "
            f"test={len(test_data.X)}"
        )

        result = train_single_head(
            model=model,
            X=train_data.X_train,
            y_main=train_data.y_main_train,
            X_validation=(
                train_data.X_validation
            ),
            y_main_validation=(
                train_data.y_main_validation
            ),
            schedule=schedule,
            device=device,
            optimizer=None,
            log_prefix=(
                f"SUPERVISED_REFERENCE:{year}"
            ),
        )

        record = evaluate_frozen_year(
            model=result.model,
            X=test_data.X,
            y_main=test_data.y_main,
            year=year,
            method_name=(
                "supervised_reference"
            ),
            device=device,
        )

        records.append(
            record
        )

        print(
            f"[SUPERVISED_REFERENCE] "
            f"year={year} | "
            f"accuracy={record.accuracy:.4f}"
        )

    return tuple(
        records
    )


def _run_final_evaluation(
    *,
    config: ExperimentConfig,
    models: FinalModels,
    protocol: V3Protocol,
    device: torch.device,
) -> FinalEvaluation:

    print()
    print("=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    id_frozen = _evaluate_id_frozen(
        model=models.single,
        protocol=protocol,
        device=device,
    )

    ood_frozen = _evaluate_ood_frozen(
        model=models.single,
        protocol=protocol,
        device=device,
    )

    tta = _evaluate_tta_methods(
        config=config,
        models=models,
        protocol=protocol,
        device=device,
    )

    supervised_reference = (
        _evaluate_supervised_reference(
            config=config,
            source_model=models.single,
            protocol=protocol,
            device=device,
        )
    )

    return FinalEvaluation(
        id_frozen=id_frozen,
        ood_frozen=ood_frozen,
        tta=tta,
        supervised_reference=(
            supervised_reference
        ),
    )

def _apply_best_overrides(
    *,
    config: ExperimentConfig,
    tuning_result: Any,
) -> ExperimentConfig:
    """
    Materialize the complete best V3 configuration.

    Order matters:

        1. model HP
        2. aux-head HP
        3. TTA-method HP

    Later overrides are method-specific and therefore must not
    alter the already selected model/aux configuration.
    """

    overrides: dict[str, Any] = {}

    model_result = getattr(
        tuning_result,
        "model",
        None,
    )

    aux_result = getattr(
        tuning_result,
        "aux_head",
        None,
    )

    method_results = getattr(
        tuning_result,
        "methods",
        None,
    )

    if model_result is None:
        raise RuntimeError(
            "V3 tuning result is missing "
            "the model stage."
        )

    if aux_result is None:
        raise RuntimeError(
            "V3 tuning result is missing "
            "the aux_head stage."
        )

    if not isinstance(
        method_results,
        dict,
    ):
        raise RuntimeError(
            "V3 tuning result has invalid "
            "method results."
        )

    overrides.update(
        model_result.best_overrides
    )

    overrides.update(
        aux_result.best_overrides
    )

    for method_name, stage_result in (
        method_results.items()
    ):

        method_overrides = getattr(
            stage_result,
            "best_overrides",
            None,
        )

        if not isinstance(
            method_overrides,
            dict,
        ):
            raise RuntimeError(
                f"Tuning result for {method_name!r} "
                "does not expose best_overrides."
            )

        overrides.update(
            method_overrides
        )

    # The Double Head main path must have exactly the same
    # selected architecture as the Single Head.
    selected_hidden_dim = overrides.get(
        "model.single_head.shared_hidden_dim",
        config.get(
            "model",
            "single_head",
            "shared_hidden_dim",
        ),
    )

    overrides[
        "model.double_head.shared_hidden_dim"
    ] = selected_hidden_dim

    print()
    print(
        "[TUNING] applying "
        f"{len(overrides)} selected final overrides"
    )

    for key in sorted(
        overrides
    ):
        print(
            f"[TUNING][BEST] "
            f"{key}={overrides[key]}"
        )

    return config.with_overrides(
        overrides
    )

def run_experiment(
    config: ExperimentConfig,
    *,
    tuning_config: Mapping[str, Any] | None = None,
    loaded_best_overrides: Mapping[
        str,
        Any,
    ] | None = None,
    run_directory: Path | None = None,
) -> ExperimentResult:
    """
    V3 experiment orchestrator.

    Modes
    -----
    tune:
        run tuning only.

    test:
        no tuning; final run uses explicitly supplied
        loaded_best_overrides (or parameters already present
        in the resolved config).

    tune_and_test:
        tune first, apply ONLY selected hyperparameters,
        then restart final training from scratch.

    Important
    ---------
    The final training function cannot receive the models
    produced during tuning. This structurally prevents tuning
    checkpoint reuse.
    """

    set_seed(
        config.experiment_seed
    )

    device = _device()

    print()
    print("=" * 80)
    print("METADATA TTA V3")
    print("=" * 80)
    print(
        f"dataset={config.dataset_name}"
    )
    print(
        f"mode={config.mode}"
    )
    print(
        f"experiment_seed="
        f"{config.experiment_seed}"
    )
    print(
        f"split_seed={config.split_seed}"
    )
    print(
        f"device={device}"
    )

    bundle = build_data_bundle(
        config
    )

    protocol = _build_protocol(
        bundle=bundle,
        config=config,
    )

    effective_config = config
    tuning_result = None

    # ========================================================
    # TUNING
    # ========================================================

    if config.mode in {
        "tune",
        "tune_and_test",
    }:

        if tuning_config is None:
            raise ValueError(
                f"mode={config.mode!r} requires "
                "tuning_config."
            )

        # Local import prevents circular import:
        # old tuning modules historically imported _device
        # from experiment.runner.
        from metadata_tta.tuning.v3 import (
            tune_v3,
        )

        tuning_output = (
            Path(run_directory)
            / "tuning"
            if run_directory is not None
            else (
                config.results_root
                / config.dataset_name
                / "tuning"
            )
        )

        tuning_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        tuning_result = tune_v3(
            base_config=config,
            tuning_config=tuning_config,
            bundle=bundle,
            protocol=protocol,
            experiment_seed=(
                config.experiment_seed
            ),
            output_root=tuning_output,
            enabled_methods=[
                method_name
                for method_name
                in TTA_METHOD_ORDER
                if config.method_enabled(
                    method_name
                )
            ],
            device=device,
        )

        if config.mode == "tune":
            return ExperimentResult(
                config=config,
                run_directory=run_directory,
                final_models=None,
                evaluation=None,
                tuning_result=tuning_result,
            )

        effective_config = (
            _apply_best_overrides(
                config=config,
                tuning_result=tuning_result,
            )
        )

        print()
        print(
            "[CHECK][PASS] tuning models will NOT "
            "be reused for final training"
        )

    # ========================================================
    # TEST-ONLY OVERRIDES
    # ========================================================

    elif config.mode == "test":

        if loaded_best_overrides:

            effective_config = (
                config.with_overrides(
                    loaded_best_overrides
                )
            )

            print(
                "[TEST] loaded explicit tuned "
                "hyperparameter overrides"
            )

        else:

            print(
                "[TEST] no external best overrides: "
                "using hyperparameters already present "
                "in resolved config"
            )

    else:
        raise RuntimeError(
            f"Unhandled mode {config.mode!r}."
        )

    # ========================================================
    # CRITICAL RESTART
    # ========================================================

    # Reset all experiment RNG state before final training.
    #
    # More importantly, _train_final_models constructs its
    # own fresh Single Head. No model object from tune_v3()
    # enters this call.
    set_seed(
        effective_config.experiment_seed
    )

    final_models = _train_final_models(
        config=effective_config,
        bundle=bundle,
        protocol=protocol,
        device=device,
    )

    evaluation = _run_final_evaluation(
        config=effective_config,
        models=final_models,
        protocol=protocol,
        device=device,
    )

    return ExperimentResult(
        config=effective_config,
        run_directory=run_directory,
        final_models=final_models,
        evaluation=evaluation,
        tuning_result=tuning_result,
    )