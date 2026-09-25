from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping

from metadata_tta.tta.metadata import (
    compute_source_aux_gradient_norms,
)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from metadata_tta.config import ExperimentConfig
from metadata_tta.data import DataBundle
from metadata_tta.evaluation import (
    evaluate_frozen_year,
    evaluate_tta_stream,
)
from metadata_tta.protocols import (
    SupervisedSlice,
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
from metadata_tta.tta import get_method_class

from .config import apply_overrides
from .results import (
    ensure_directory,
    save_best_result,
    save_trial_rows,
)


METADATA_METHODS = {
    "metadata_episodic",
    "metadata_cumulative",
    "metadata_cumulative_annual_reset",
    "metadata_cumulative_drift_reset",
    "metadata_cumulative_drift_annual_reset",
    "temporal_gradient_ema",
}

IMPLEMENTED_TUNABLE_METHODS = (
    "metadata_episodic",
    "metadata_cumulative",
    "metadata_cumulative_annual_reset",
    "metadata_cumulative_drift_reset",
    "metadata_cumulative_drift_annual_reset",
    "temporal_gradient_ema",
    "tent",
)


@dataclass(frozen=True)
class StageResult:
    name: str
    best_score: float
    best_parameters: dict[str, Any]
    best_overrides: dict[str, Any]
    output_directory: Path


@dataclass(frozen=True)
class V3TuningResult:
    model: StageResult
    aux_head: StageResult
    aux_alignment: StageResult
    methods: dict[str, StageResult]

    model_config: ExperimentConfig
    aux_config: ExperimentConfig

    source_single: nn.Module
    source_double: nn.Module


def _experiment_config(
    data: Mapping[str, Any],
) -> ExperimentConfig:
    return ExperimentConfig(
        copy.deepcopy(
            dict(data)
        )
    )


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def _explicit_candidates(
    section: Mapping[str, Any],
    *,
    stage_name: str,
) -> list[dict[str, Any]]:

    raw = section.get(
        "candidates"
    )

    if not isinstance(
        raw,
        list,
    ):
        raise ValueError(
            f"tuning.{stage_name}.candidates "
            "must be a list."
        )

    if not raw:
        raise ValueError(
            f"tuning.{stage_name}.candidates "
            "cannot be empty."
        )

    candidates = []

    for index, candidate in enumerate(
        raw
    ):
        if not isinstance(
            candidate,
            Mapping,
        ):
            raise ValueError(
                f"{stage_name}.candidates[{index}] "
                "must be a mapping."
            )

        candidate = dict(
            candidate
        )

        if "id" not in candidate:
            candidate["id"] = (
                f"{stage_name}_{index:02d}"
            )

        candidates.append(
            candidate
        )

    return candidates

def _expand_gradient_norm_quantile_candidates(
    *,
    method_name: str,
    tuning_section: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    source_gradient_norms: np.ndarray | None,
) -> list[dict[str, Any]]:
    """
    Expand metadata candidates over source-calibrated
    gradient-norm quantiles.

    The tunable hyperparameter is the dimensionless quantile q.

    For q != None:

        threshold = quantile(
            source-validation gradient norms,
            q,
        )

    The resulting numerical threshold is stored in
    gradient_norm_max and used by MetadataTTA.

    None means: gradient-norm gate disabled.
    """

    raw_values = tuning_section.get(
        "gradient_norm_quantile_values"
    )

    if raw_values is None:
        return candidates

    if method_name not in {
        "metadata_episodic",
        "metadata_cumulative",
    }:
        raise ValueError(
            "gradient_norm_quantile_values may only be "
            "configured for metadata_episodic or "
            "metadata_cumulative."
        )

    if not isinstance(
        raw_values,
        list,
    ):
        raise ValueError(
            f"tuning.{method_name}."
            "gradient_norm_quantile_values must be a list."
        )

    if not raw_values:
        raise ValueError(
            f"tuning.{method_name}."
            "gradient_norm_quantile_values cannot be empty."
        )

    if source_gradient_norms is None:
        raise RuntimeError(
            f"{method_name}: source gradient-norm calibration "
            "is required for quantile tuning."
        )

    calibration = np.asarray(
        source_gradient_norms,
        dtype=np.float64,
    )

    if (
        calibration.ndim != 1
        or calibration.size == 0
    ):
        raise RuntimeError(
            "Source gradient-norm calibration must be "
            "a non-empty 1D array."
        )

    if not np.all(
        np.isfinite(
            calibration
        )
    ):
        raise RuntimeError(
            "Source gradient-norm calibration contains "
            "non-finite values."
        )

    if np.any(
        calibration < 0.0
    ):
        raise RuntimeError(
            "Source gradient norms cannot be negative."
        )

    expanded: list[
        dict[str, Any]
    ] = []

    for candidate in candidates:

        base_parameters = dict(
            candidate.get(
                "parameters",
                {},
            )
        )

        base_overrides = dict(
            candidate.get(
                "overrides",
                {},
            )
        )

        for raw_value in raw_values:

            quantile = (
                None
                if raw_value is None
                else float(
                    raw_value
                )
            )

            if (
                quantile is not None
                and not (
                    0.0
                    < quantile
                    < 1.0
                )
            ):
                raise ValueError(
                    "gradient_norm_quantile must satisfy "
                    "0 < q < 1."
                )

            if quantile is None:

                calibrated_threshold = None

                suffix = "off"

            else:

                calibrated_threshold = float(
                    np.quantile(
                        calibration,
                        quantile,
                    )
                )

                if calibrated_threshold <= 0.0:
                    raise RuntimeError(
                        "Source-calibrated gradient threshold "
                        "must be > 0."
                    )

                suffix = (
                    str(
                        quantile
                    )
                    .replace(
                        ".",
                        "p",
                    )
                )

            parameters = dict(
                base_parameters
            )

            parameters[
                "gradient_norm_quantile"
            ] = quantile

            parameters[
                "calibrated_gradient_norm_max"
            ] = calibrated_threshold

            overrides = dict(
                base_overrides
            )

            overrides[
                (
                    f"methods.{method_name}."
                    "gradient_norm_quantile"
                )
            ] = quantile

            overrides[
                (
                    f"methods.{method_name}."
                    "gradient_norm_max"
                )
            ] = calibrated_threshold

            expanded.append(
                {
                    "id":
                        (
                            f"{candidate['id']}"
                            f"_q{suffix}"
                        ),

                    "parameters":
                        parameters,

                    "overrides":
                        overrides,
                }
            )

    return expanded

def _candidate_overrides(
    candidate: Mapping[str, Any],
) -> dict[str, Any]:

    overrides = candidate.get(
        "overrides"
    )

    if not isinstance(
        overrides,
        Mapping,
    ):
        raise ValueError(
            f"Candidate {candidate.get('id')} "
            "requires an overrides mapping."
        )

    return {
        str(path): value
        for path, value
        in overrides.items()
    }


def _candidate_parameters(
    candidate: Mapping[str, Any],
) -> dict[str, Any]:

    parameters = candidate.get(
        "parameters",
        {}
    )

    if not isinstance(
        parameters,
        Mapping,
    ):
        raise ValueError(
            f"Candidate {candidate.get('id')} "
            "parameters must be a mapping."
        )

    return dict(
        parameters
    )


def _train_temporal_single(
    *,
    config: ExperimentConfig,
    bundle: DataBundle,
    source_slices: tuple[SupervisedSlice, ...],
    experiment_seed: int,
    device: torch.device,
    log_prefix: str,
) -> nn.Module:

    if not source_slices:
        raise ValueError(
            "At least one source year is required."
        )

    set_seed(
        experiment_seed
    )

    model = create_single_head_model(
        input_dim=bundle.feature_dim,
        config=config,
    ).to(device)

    set_seed(
        experiment_seed
    )

    optimizer = None

    for index, data in enumerate(
        source_slices
    ):

        schedule = build_training_schedule(
            config=config,
            initialized_from_previous=(
                index > 0
            ),
            model_kind="single_head",
        )

        result = train_single_head(
            model=model,
            X=data.X_train,
            y_main=data.y_main_train,
            schedule=schedule,
            device=device,
            optimizer=optimizer,
            X_validation=(
                data.X_validation
            ),
            y_main_validation=(
                data.y_main_validation
            ),
            log_prefix=(
                f"{log_prefix} | "
                f"year={data.year}"
            ),
        )

        model = result.model
        optimizer = result.optimizer

    model.eval()

    return model


def _mean_source_validation_accuracy(
    *,
    model: nn.Module,
    source_slices: tuple[SupervisedSlice, ...],
    device: torch.device,
) -> tuple[
    float,
    dict[int, float],
]:

    yearly = {}

    for data in source_slices:

        record = evaluate_frozen_year(
            model=model,
            X=data.X_validation,
            y_main=data.y_main_validation,
            year=data.year,
            method_name="model_validation",
            device=device,
        )

        yearly[
            int(data.year)
        ] = float(
            record.accuracy
        )

    return (
        float(
            mean(
                yearly.values()
            )
        ),
        yearly,
    )


def tune_model(
    *,
    base_config: ExperimentConfig,
    tuning_section: Mapping[str, Any],
    bundle: DataBundle,
    protocol: V3Protocol,
    experiment_seed: int,
    output_root: Path,
    device: torch.device,
) -> tuple[
    StageResult,
    nn.Module,
]:

    candidates = _explicit_candidates(
        tuning_section,
        stage_name="model",
    )

    output_directory = ensure_directory(
        output_root / "model"
    )

    source_slices = (
        protocol.pseudo_source_supervised()
    )

    rows = []

    best_score = float("-inf")
    best_parameters = {}
    best_overrides = {}

    best_model: nn.Module | None = None

    print(
        f"[TUNING][MODEL] "
        f"candidates={len(candidates)}"
    )

    for trial_index, candidate in enumerate(
        candidates,
        start=1,
    ):

        parameters = _candidate_parameters(
            candidate
        )

        overrides = _candidate_overrides(
            candidate
        )

        trial_config = apply_overrides(
            config=base_config,
            overrides=overrides,
        )

        print(
            f"[TUNING][MODEL] "
            f"trial={trial_index}/{len(candidates)} "
            f"id={candidate['id']} "
            f"params={parameters}"
        )

        model = _train_temporal_single(
            config=trial_config,
            bundle=bundle,
            source_slices=source_slices,
            experiment_seed=experiment_seed,
            device=device,
            log_prefix=(
                f"MODEL {candidate['id']}"
            ),
        )

        score, yearly = (
            _mean_source_validation_accuracy(
                model=model,
                source_slices=source_slices,
                device=device,
            )
        )

        row = {
            "trial_id":
                candidate["id"],

            "mean_validation_accuracy":
                score,
        }

        for key, value in (
            parameters.items()
        ):
            row[
                f"param_{key}"
            ] = value

        for year, value in (
            yearly.items()
        ):
            row[
                f"val_accuracy_{year}"
            ] = value

        rows.append(
            row
        )

        save_trial_rows(
            output_directory / "trials.csv",
            rows,
        )

        print(
            f"[TUNING][MODEL] "
            f"id={candidate['id']} "
            f"mean_val_accuracy={score:.6f}"
        )

        if score > best_score:

            best_score = score
            best_parameters = parameters
            best_overrides = overrides

            best_model = copy.deepcopy(
                model
            ).to(
                device
            )

            best_model.eval()

    if best_model is None:
        raise RuntimeError(
            "Model tuning did not produce "
            "a best model."
        )

    save_best_result(
        output_directory / "best.yaml",
        kind="model",
        name="model",
        score_name=(
            "mean_validation_accuracy"
        ),
        score=best_score,
        parameters=best_parameters,
        overrides=best_overrides,
        extra={
            "years":
                list(
                    protocol.years.pseudo_source
                ),

            "experiment_seed":
                int(
                    experiment_seed
                ),

            "split_seed":
                int(
                    protocol.split_seed
                ),
        },
    )

    print(
        f"[TUNING][MODEL][BEST] "
        f"score={best_score:.6f} "
        f"params={best_parameters}"
    )

    stage_result = StageResult(
        name="model",
        best_score=best_score,
        best_parameters=best_parameters,
        best_overrides=best_overrides,
        output_directory=output_directory,
    )

    return (
        stage_result,
        best_model,
    )


def _build_double_from_single(
    *,
    single_model: nn.Module,
    config: ExperimentConfig,
    bundle: DataBundle,
    experiment_seed: int,
    device: torch.device,
) -> nn.Module:

    set_seed(
        experiment_seed
    )

    double_model = (
        create_double_head_model(
            input_dim=bundle.feature_dim,
            config=config,
        )
        .to(device)
    )

    double_model = (
        initialize_double_from_single(
            single_model=single_model,
            double_model=double_model,
        )
    )

    double_model.eval()

    return double_model


def _assert_main_equivalence(
    *,
    single_model: nn.Module,
    double_model: nn.Module,
    X: np.ndarray,
    device: torch.device,
) -> None:

    if len(X) == 0:
        raise RuntimeError(
            "Cannot verify model equivalence "
            "on an empty array."
        )

    sample = torch.as_tensor(
        X[
            : min(
                64,
                len(X),
            )
        ],
        dtype=torch.float32,
        device=device,
    )

    single_model.eval()
    double_model.eval()

    with torch.no_grad():

        single_logits = (
            single_model(
                sample
            )
        )

        double_logits = (
            double_model(
                sample
            )
        )

    if not torch.equal(
        single_logits,
        double_logits,
    ):

        max_error = float(
            torch.max(
                torch.abs(
                    single_logits
                    - double_logits
                )
            ).item()
        )

        raise RuntimeError(
            "Single/Double main-path equivalence "
            "check failed before Aux training. "
            f"max_abs_error={max_error:.12e}"
        )


def _concatenate_aux_source(
    source_slices: tuple[
        SupervisedSlice,
        ...
    ],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    X_train = np.concatenate(
        [
            data.X_train
            for data in source_slices
        ],
        axis=0,
    )

    y_main_train = np.concatenate(
        [
            data.y_main_train
            for data in source_slices
        ],
        axis=0,
    )

    y_aux_train = np.concatenate(
        [
            data.y_aux_train
            for data in source_slices
        ],
        axis=0,
    )

    X_validation = np.concatenate(
        [
            data.X_validation
            for data in source_slices
        ],
        axis=0,
    )

    y_main_validation = np.concatenate(
        [
            data.y_main_validation
            for data in source_slices
        ],
        axis=0,
    )

    y_aux_validation = np.concatenate(
        [
            data.y_aux_validation
            for data in source_slices
        ],
        axis=0,
    )

    return (
        X_train,
        y_main_train,
        y_aux_train,
        X_validation,
        y_main_validation,
        y_aux_validation,
    )


def _aux_validation_metrics(
    *,
    model: nn.Module,
    X: np.ndarray,
    y_aux: np.ndarray,
    device: torch.device,
) -> tuple[
    float,
    float,
]:

    model.eval()

    X_tensor = torch.as_tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    y_tensor = torch.as_tensor(
        y_aux,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():

        logits = model.forward_aux(
            X_tensor
        )

        loss = F.cross_entropy(
            logits,
            y_tensor,
        )

        accuracy = (
            logits.argmax(
                dim=1
            )
            == y_tensor
        ).float().mean()

    return (
        float(
            loss.item()
        ),
        float(
            accuracy.item()
        ),
    )


def _train_aux_from_fixed_single(
    *,
    source_single: nn.Module,
    config: ExperimentConfig,
    bundle: DataBundle,
    protocol: V3Protocol,
    experiment_seed: int,
    device: torch.device,
    log_prefix: str,
) -> nn.Module:
    """
    Build a fresh Double Head from the SAME frozen source Single
    and train ONLY the auxiliary head.

    The source Single is never retrained and never optimized here.
    """

    source_slices = (
        protocol.pseudo_source_supervised()
    )

    (
        X_train,
        y_main_train,
        y_aux_train,
        X_validation,
        y_main_validation,
        y_aux_validation,
    ) = _concatenate_aux_source(
        source_slices
    )

    double_model = (
        _build_double_from_single(
            single_model=source_single,
            config=config,
            bundle=bundle,
            experiment_seed=experiment_seed,
            device=device,
        )
    )

    _assert_main_equivalence(
        single_model=source_single,
        double_model=double_model,
        X=X_validation,
        device=device,
    )

    training = config.section(
        "training"
    )

    base_training = training[
        "base"
    ]

    aux_training = training.get(
        "double_head",
        {},
    )

    early_stopping = (
        aux_training.get(
            "early_stopping",
            base_training.get(
                "early_stopping",
                {},
            ),
        )
    )

    learning_rate = float(
        aux_training.get(
            "learning_rate",
            base_training[
                "learning_rate"
            ],
        )
    )

    weight_decay = float(
        aux_training.get(
            "weight_decay",
            0.0,
        )
    )

    batch_size = int(
        aux_training.get(
            "batch_size",
            base_training[
                "batch_size"
            ],
        )
    )

    epochs = int(
        aux_training.get(
            "epochs",
            base_training[
                "epochs"
            ],
        )
    )

    gradient_alignment_weight = float(
        aux_training.get(
            "gradient_alignment_weight",
            0.0,
        )
    )

    gradient_alignment_epsilon = float(
        aux_training.get(
            "gradient_alignment_epsilon",
            1.0e-8,
        )
    )

    set_seed(
        experiment_seed
    )

    double_model = train_aux_head_only(
        model=double_model,

        X=X_train,
        y_main=y_main_train,
        y_aux=y_aux_train,

        X_validation=X_validation,
        y_main_validation=(
            y_main_validation
        ),
        y_aux_validation=(
            y_aux_validation
        ),

        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        epochs=epochs,

        patience=int(
            early_stopping.get(
                "patience",
                5,
            )
        ),

        min_delta=float(
            early_stopping.get(
                "min_delta",
                0.0,
            )
        ),

        gradient_alignment_weight=(
            gradient_alignment_weight
        ),

        gradient_alignment_epsilon=(
            gradient_alignment_epsilon
        ),

        device=device,
        log_prefix=log_prefix,
    )

    _assert_main_equivalence(
        single_model=source_single,
        double_model=double_model,
        X=X_validation,
        device=device,
    )

    source_single.eval()
    double_model.eval()

    return double_model


def tune_aux_head(
    *,
    source_single: nn.Module,
    model_config: ExperimentConfig,
    tuning_section: Mapping[str, Any],
    bundle: DataBundle,
    protocol: V3Protocol,
    experiment_seed: int,
    output_root: Path,
    device: torch.device,
) -> StageResult:

    candidates = _explicit_candidates(
        tuning_section,
        stage_name="aux_head",
    )

    output_directory = ensure_directory(
        output_root
        / "aux_head"
    )

    source_slices = (
        protocol.pseudo_source_supervised()
    )

    (
        _,
        _,
        _,
        X_validation,
        _,
        y_aux_validation,
    ) = _concatenate_aux_source(
        source_slices
    )

    rows = []

    best_score = float(
        "-inf"
    )

    best_parameters = {}
    best_overrides = {}

    print(
        f"[TUNING][AUX] "
        f"candidates={len(candidates)}"
    )

    for trial_index, candidate in enumerate(
        candidates,
        start=1,
    ):

        parameters = _candidate_parameters(
            candidate
        )

        overrides = _candidate_overrides(
            candidate
        )

        trial_config = apply_overrides(
            config=model_config,
            overrides=overrides,
        )

        print(
            f"[TUNING][AUX] "
            f"trial={trial_index}/{len(candidates)} "
            f"id={candidate['id']} "
            f"params={parameters}"
        )

        double_model = (
            _train_aux_from_fixed_single(
                source_single=source_single,
                config=trial_config,
                bundle=bundle,
                protocol=protocol,
                experiment_seed=experiment_seed,
                device=device,
                log_prefix=(
                    f"AUX {candidate['id']}"
                ),
            )
        )

        (
            validation_loss,
            validation_accuracy,
        ) = _aux_validation_metrics(
            model=double_model,
            X=X_validation,
            y_aux=y_aux_validation,
            device=device,
        )

        score = validation_accuracy

        rows.append(
            {
                "trial_id":
                    candidate["id"],

                "validation_aux_accuracy":
                    validation_accuracy,

                "validation_aux_loss":
                    validation_loss,

                **{
                    f"param_{key}":
                        value
                    for key, value
                    in parameters.items()
                },
            }
        )

        save_trial_rows(
            output_directory
            / "trials.csv",
            rows,
        )

        print(
            f"[TUNING][AUX] "
            f"id={candidate['id']} "
            f"val_aux_accuracy="
            f"{validation_accuracy:.6f} "
            f"val_aux_loss="
            f"{validation_loss:.6f}"
        )

        if score > best_score:

            best_score = score
            best_parameters = parameters
            best_overrides = overrides

    save_best_result(
        output_directory
        / "best.yaml",
        kind="aux_head",
        name="aux_head",
        score_name=(
            "validation_aux_accuracy"
        ),
        score=best_score,
        parameters=best_parameters,
        overrides=best_overrides,
        extra={
            "years":
                list(
                    protocol.years.pseudo_source
                ),

            "experiment_seed":
                int(
                    experiment_seed
                ),

            "split_seed":
                int(
                    protocol.split_seed
                ),

            "source_main_reused":
                True,
        },
    )

    print(
        f"[TUNING][AUX][BEST] "
        f"score={best_score:.6f} "
        f"params={best_parameters}"
    )

    return StageResult(
        name="aux_head",
        best_score=best_score,
        best_parameters=best_parameters,
        best_overrides=best_overrides,
        output_directory=output_directory,
    )


def tune_aux_alignment(
    *,
    source_single: nn.Module,
    base_config: ExperimentConfig,
    tuning_section: Mapping[str, Any],
    episodic_section: Mapping[str, Any],
    bundle: DataBundle,
    protocol: V3Protocol,
    experiment_seed: int,
    output_root: Path,
    device: torch.device,
) -> StageResult:
    """
    Tune gradient_alignment_weight using pseudo-OOD Episodic
    performance.

    Every alignment candidate starts from the exact SAME
    pseudo-source Single model.

    Only a fresh Aux head is trained for each candidate.

    gradient_norm_max is forced OFF during this stage.
    """

    candidates = _explicit_candidates(
        tuning_section,
        stage_name="aux_alignment",
    )

    episodic_candidates = _explicit_candidates(
        episodic_section,
        stage_name="metadata_episodic",
    )

    probe_candidates = []

    for candidate in episodic_candidates:

        parameters = dict(
            candidate.get(
                "parameters",
                {},
            )
        )

        overrides = dict(
            candidate.get(
                "overrides",
                {},
            )
        )

        overrides[
            "methods.metadata_episodic.gradient_norm_max"
        ] = None

        overrides[
            "methods.metadata_episodic.gradient_norm_quantile"
        ] = None

        probe_candidates.append(
            {
                "id":
                    candidate["id"],

                "parameters":
                    parameters,

                "overrides":
                    overrides,
            }
        )

    probe_section = {
        "candidates":
            probe_candidates
    }

    output_directory = ensure_directory(
        output_root
        / "aux_alignment"
    )

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

    rows = []

    for trial_index, candidate in enumerate(
        candidates,
        start=1,
    ):

        parameters = _candidate_parameters(
            candidate
        )

        overrides = _candidate_overrides(
            candidate
        )

        trial_config = apply_overrides(
            config=base_config,
            overrides=overrides,
        )

        print(
            f"[TUNING][AUX-ALIGN] "
            f"trial={trial_index}/{len(candidates)} "
            f"id={candidate['id']} "
            f"params={parameters}"
        )

        # IMPORTANT:
        # same source_single for every lambda.
        # Only Aux head is freshly initialized/trained.
        source_double = (
            _train_aux_from_fixed_single(
                source_single=source_single,
                config=trial_config,
                bundle=bundle,
                protocol=protocol,
                experiment_seed=experiment_seed,
                device=device,
                log_prefix=(
                    f"AUX-ALIGN "
                    f"{candidate['id']}"
                ),
            )
        )

        probe_result = tune_tta_method(
            method_name="metadata_episodic",
            base_config=trial_config,
            tuning_section=probe_section,
            inherited_overrides={},
            source_single=source_single,
            source_double=source_double,
            protocol=protocol,
            experiment_seed=experiment_seed,
            output_root=(
                output_directory
                / candidate["id"]
            ),
            device=device,
        )

        score = float(
            probe_result.best_score
        )

        row = {
            "trial_id":
                candidate["id"],

            "probe_mean_delta_accuracy":
                score,

            "probe_best_parameters":
                str(
                    probe_result.best_parameters
                ),
        }

        for key, value in (
            parameters.items()
        ):
            row[
                f"param_{key}"
            ] = value

        rows.append(
            row
        )

        save_trial_rows(
            output_directory
            / "trials.csv",
            rows,
        )

        print(
            f"[TUNING][AUX-ALIGN] "
            f"id={candidate['id']} "
            f"probe_mean_delta="
            f"{score:+.6f}"
        )

        if score > best_score:

            best_score = score
            best_parameters = parameters
            best_overrides = overrides

    save_best_result(
        output_directory
        / "best.yaml",
        kind="aux_alignment",
        name="aux_alignment",
        score_name=(
            "episodic_probe_mean_delta_accuracy"
        ),
        score=best_score,
        parameters=best_parameters,
        overrides=best_overrides,
        extra={
            "probe_method":
                "metadata_episodic",

            "probe_gradient_norm_gate":
                "disabled",

            "stream_years":
                list(
                    protocol.years.pseudo_ood
                ),

            "stream_split":
                "validation",

            "experiment_seed":
                int(
                    experiment_seed
                ),

            "split_seed":
                int(
                    protocol.split_seed
                ),

            "source_main_reused":
                True,
        },
    )

    print(
        "[TUNING][AUX-ALIGN][BEST] "
        f"score={best_score:+.6f} "
        f"params={best_parameters}"
    )

    return StageResult(
        name="aux_alignment",
        best_score=best_score,
        best_parameters=best_parameters,
        best_overrides=best_overrides,
        output_directory=output_directory,
    )


def train_tuning_source_models(
    *,
    config: ExperimentConfig,
    bundle: DataBundle,
    protocol: V3Protocol,
    experiment_seed: int,
    device: torch.device,
) -> tuple[
    nn.Module,
    nn.Module,
]:
    """
    Standalone helper retained for diagnostics/scripts.

    tune_v3 itself does NOT call this function anymore because
    it already owns the selected pseudo-source Single model.
    """

    source_slices = (
        protocol.pseudo_source_supervised()
    )

    source_single = _train_temporal_single(
        config=config,
        bundle=bundle,
        source_slices=source_slices,
        experiment_seed=experiment_seed,
        device=device,
        log_prefix="TTA SOURCE",
    )

    source_double = (
        _train_aux_from_fixed_single(
            source_single=source_single,
            config=config,
            bundle=bundle,
            protocol=protocol,
            experiment_seed=experiment_seed,
            device=device,
            log_prefix="TTA SOURCE AUX",
        )
    )

    source_single.eval()
    source_double.eval()

    return (
        source_single,
        source_double,
    )


def _frozen_reference(
    *,
    model: nn.Module,
    stream,
    device: torch.device,
) -> dict[int, float]:

    result = {}

    for data in stream:

        record = evaluate_frozen_year(
            model=model,
            X=data.X,
            y_main=data.y_main,
            year=data.year,
            method_name="frozen",
            device=device,
        )

        result[
            int(data.year)
        ] = float(
            record.accuracy
        )

    return result


def _method_source_model(
    method_name: str,
    *,
    source_single: nn.Module,
    source_double: nn.Module,
) -> nn.Module:

    if method_name == "tent":
        return source_single

    if method_name in METADATA_METHODS:
        return source_double

    raise ValueError(
        f"Unsupported V3 method: "
        f"{method_name}"
    )


def _method_config(
    config: ExperimentConfig,
    method_name: str,
) -> Mapping[str, Any]:

    methods = config.section(
        "methods"
    )

    method_config = methods.get(
        method_name
    )

    if not isinstance(
        method_config,
        Mapping,
    ):
        raise KeyError(
            f"Missing methods.{method_name} "
            "configuration."
        )

    return method_config


def tune_tta_method(
    *,
    method_name: str,
    base_config: ExperimentConfig,
    tuning_section: Mapping[str, Any],
    inherited_overrides: Mapping[str, Any],
    source_single: nn.Module,
    source_double: nn.Module,
    protocol: V3Protocol,
    experiment_seed: int,
    output_root: Path,
    device: torch.device,
    source_gradient_norms: np.ndarray | None = None,
) -> StageResult:

    candidates = _explicit_candidates(
        tuning_section,
        stage_name=method_name,
    )

    candidates = (
        _expand_gradient_norm_quantile_candidates(
            method_name=method_name,
            tuning_section=tuning_section,
            candidates=candidates,
            source_gradient_norms=(
                source_gradient_norms
            ),
        )
    )

    output_directory = ensure_directory(
        output_root
        / method_name
    )

    stream_slices = (
        protocol.pseudo_ood_validation_stream()
    )

    stream = protocol.evaluator_stream(
        stream_slices
    )

    source_model = _method_source_model(
        method_name,
        source_single=source_single,
        source_double=source_double,
    )

    frozen = _frozen_reference(
        model=source_model,
        stream=stream_slices,
        device=device,
    )

    method_class = get_method_class(
        method_name
    )

    rows = []

    best_score = float(
        "-inf"
    )

    best_parameters = {}
    best_overrides = {}

    print(
        f"[TUNING][TTA][{method_name}] "
        f"candidates={len(candidates)}"
    )

    for trial_index, candidate in enumerate(
        candidates,
        start=1,
    ):

        parameters = _candidate_parameters(
            candidate
        )

        local_overrides = (
            _candidate_overrides(
                candidate
            )
        )

        complete_overrides = {
            **dict(
                inherited_overrides
            ),
            **local_overrides,
        }

        trial_config = apply_overrides(
            config=base_config,
            overrides=complete_overrides,
        )

        set_seed(
            experiment_seed
        )

        method = method_class(
            source_model=copy.deepcopy(
                source_model
            ),
            config=_method_config(
                trial_config,
                method_name,
            ),
            device=device,
        )

        print(
            f"[TUNING][TTA][{method_name}] "
            f"trial={trial_index}/{len(candidates)} "
            f"id={candidate['id']} "
            f"params={parameters}"
        )

        evaluation = evaluate_tta_stream(
            method=method,
            years=stream,
        )

        tta_by_year = {
            int(record.year):
                float(
                    record.accuracy
                )
            for record
            in evaluation.records
        }

        if set(
            tta_by_year
        ) != set(
            frozen
        ):
            raise RuntimeError(
                f"{method_name}: evaluator returned "
                "unexpected pseudo-OOD years."
            )

        deltas = {
            year:
                (
                    tta_by_year[
                        year
                    ]
                    - frozen[
                        year
                    ]
                )
            for year
            in tta_by_year
        }

        delta_values = list(
            deltas.values()
        )

        score = float(
            mean(
                delta_values
            )
        )

        row = {
            "trial_id":
                candidate["id"],

            "mean_delta_accuracy":
                score,

            "std_delta_accuracy":
                float(
                    pstdev(
                        delta_values
                    )
                    if len(
                        delta_values
                    ) > 1
                    else 0.0
                ),

            "worst_delta_accuracy":
                float(
                    min(
                        delta_values
                    )
                ),

            "n_improved_years":
                int(
                    sum(
                        delta > 0.0
                        for delta
                        in delta_values
                    )
                ),
        }

        for key, value in (
            parameters.items()
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

        for year in sorted(
            tta_by_year
        ):

            row[
                f"frozen_accuracy_{year}"
            ] = frozen[
                year
            ]

            row[
                f"tta_accuracy_{year}"
            ] = tta_by_year[
                year
            ]

            row[
                f"delta_accuracy_{year}"
            ] = deltas[
                year
            ]

        rows.append(
            row
        )

        save_trial_rows(
            output_directory
            / "trials.csv",
            rows,
        )

        print(
            f"[TUNING][TTA][{method_name}] "
            f"id={candidate['id']} "
            f"mean_delta={score:+.6f}"
        )

        if score > best_score:

            best_score = score
            best_parameters = parameters
            best_overrides = (
                complete_overrides
            )

    save_best_result(
        output_directory
        / "best.yaml",
        kind="tta",
        name=method_name,
        score_name=(
            "mean_delta_accuracy"
        ),
        score=best_score,
        parameters=best_parameters,
        overrides=best_overrides,
        extra={
            "stream_years":
                list(
                    protocol.years.pseudo_ood
                ),

            "stream_split":
                "validation",

            "experiment_seed":
                int(
                    experiment_seed
                ),

            "split_seed":
                int(
                    protocol.split_seed
                ),
        },
    )

    print(
        f"[TUNING][TTA][{method_name}][BEST] "
        f"score={best_score:+.6f} "
        f"params={best_parameters}"
    )

    return StageResult(
        name=method_name,
        best_score=best_score,
        best_parameters=best_parameters,
        best_overrides=best_overrides,
        output_directory=output_directory,
    )


def _inherit_metadata_core(
    result: StageResult,
    *,
    target_method: str,
    include_gradient_norm_gate: bool = True,
) -> dict[str, Any]:

    required = (
        "lr",
        "normalized_aux_loss_min",
        "normalized_aux_loss_max",
    )

    missing = [
        name
        for name in required
        if name not in result.best_parameters
    ]

    if missing:
        raise RuntimeError(
            "Cannot inherit cumulative metadata "
            f"hyperparameters; missing {missing}."
        )

    prefix = (
        f"methods.{target_method}"
    )

    overrides: dict[
        str,
        Any,
    ] = {
        f"{prefix}.learning_rate":
            float(
                result.best_parameters[
                    "lr"
                ]
            ),

        f"{prefix}.normalized_aux_loss_min":
            float(
                result.best_parameters[
                    "normalized_aux_loss_min"
                ]
            ),

        f"{prefix}.normalized_aux_loss_max":
            float(
                result.best_parameters[
                    "normalized_aux_loss_max"
                ]
            ),
    }

    if include_gradient_norm_gate:

        required_gate_parameters = (
            "gradient_norm_quantile",
            "calibrated_gradient_norm_max",
        )

        missing_gate_parameters = [
            name
            for name in required_gate_parameters
            if name not in result.best_parameters
        ]

        if missing_gate_parameters:
            raise RuntimeError(
                "Cannot inherit gradient-norm gate; "
                f"missing {missing_gate_parameters}."
            )

        raw_quantile = (
            result.best_parameters[
                "gradient_norm_quantile"
            ]
        )

        raw_threshold = (
            result.best_parameters[
                "calibrated_gradient_norm_max"
            ]
        )

        gradient_norm_quantile = (
            None
            if raw_quantile is None
            else float(
                raw_quantile
            )
        )

        gradient_norm_max = (
            None
            if raw_threshold is None
            else float(
                raw_threshold
            )
        )

        if (
            gradient_norm_quantile is None
        ) != (
            gradient_norm_max is None
        ):
            raise RuntimeError(
                "Inherited gradient-norm quantile and "
                "calibrated threshold are inconsistent."
            )

        overrides[
            f"{prefix}.gradient_norm_quantile"
        ] = gradient_norm_quantile

        overrides[
            f"{prefix}.gradient_norm_max"
        ] = gradient_norm_max

    return overrides


def _inherit_drift_configuration(
    cumulative_result: StageResult,
    drift_result: StageResult,
    *,
    target_method: str,
) -> dict[str, Any]:

    overrides = _inherit_metadata_core(
        cumulative_result,
        target_method=target_method,
    )

    if (
        "adwin_delta"
        not in drift_result.best_parameters
    ):
        raise RuntimeError(
            "Cannot inherit drift configuration: "
            "best adwin_delta is missing."
        )

    prefix = (
        f"methods.{target_method}"
    )

    overrides.update(
        {
            f"{prefix}.adwin_delta":
                float(
                    drift_result.best_parameters[
                        "adwin_delta"
                    ]
                ),

            f"{prefix}.drift_signal":
                "normalized_aux_loss",
        }
    )

    return overrides


def tune_v3(
    *,
    base_config: ExperimentConfig,
    tuning_config: Mapping[str, Any],
    bundle: DataBundle,
    protocol: V3Protocol,
    experiment_seed: int,
    output_root: Path,
    enabled_methods: list[str],
    device: torch.device | None = None,
) -> V3TuningResult:

    if device is None:
        device = _device()

    tuning_root = tuning_config.get(
        "tuning",
        tuning_config,
    )

    if not isinstance(
        tuning_root,
        Mapping,
    ):
        raise ValueError(
            "Tuning configuration must "
            "be a mapping."
        )

    print()
    print("=" * 80)
    print("V3 TUNING")
    print(
        f"experiment_seed="
        f"{experiment_seed} | "
        f"split_seed="
        f"{protocol.split_seed} | "
        f"device={device}"
    )
    print(
        "pseudo_source="
        f"{list(protocol.years.pseudo_source)}"
    )
    print(
        "pseudo_ood(validation only)="
        f"{list(protocol.years.pseudo_ood)}"
    )
    print("=" * 80)

    # ========================================================
    # 1. MODEL
    #
    # This is the ONLY place where pseudo-source Single
    # training happens inside tune_v3.
    # ========================================================

    model_section = tuning_root.get(
        "model"
    )

    if not isinstance(
        model_section,
        Mapping,
    ):
        raise ValueError(
            "Missing tuning.model section."
        )

    (
        model_result,
        source_single,
    ) = tune_model(
        base_config=base_config,
        tuning_section=model_section,
        bundle=bundle,
        protocol=protocol,
        experiment_seed=experiment_seed,
        output_root=output_root,
        device=device,
    )

    model_config = apply_overrides(
        config=base_config,
        overrides=(
            model_result.best_overrides
        ),
    )

    model_config = apply_overrides(
        config=model_config,
        overrides={
            "model.double_head.shared_hidden_dim":
                model_config.get(
                    "model",
                    "single_head",
                    "shared_hidden_dim",
                ),

            "model.double_head.dropout":
                model_config.get(
                    "model",
                    "single_head",
                    "dropout",
                ),
        },
    )

    source_single.eval()

    # ========================================================
    # 2. AUX HEAD LR
    #
    # SAME source_single for every Aux candidate.
    # ========================================================

    aux_section = tuning_root.get(
        "aux_head"
    )

    if not isinstance(
        aux_section,
        Mapping,
    ):
        raise ValueError(
            "Missing tuning.aux_head section."
        )

    aux_result = tune_aux_head(
        source_single=source_single,
        model_config=model_config,
        tuning_section=aux_section,
        bundle=bundle,
        protocol=protocol,
        experiment_seed=experiment_seed,
        output_root=output_root,
        device=device,
    )

    aux_config = apply_overrides(
        config=model_config,
        overrides=(
            aux_result.best_overrides
        ),
    )

    # ========================================================
    # 3. AUX GRADIENT ALIGNMENT
    #
    # SAME source_single for every lambda.
    # ========================================================

    alignment_section = tuning_root.get(
        "aux_alignment"
    )

    if not isinstance(
        alignment_section,
        Mapping,
    ):
        raise ValueError(
            "Missing tuning.aux_alignment section."
        )

    episodic_section = tuning_root.get(
        "metadata_episodic"
    )

    if not isinstance(
        episodic_section,
        Mapping,
    ):
        raise ValueError(
            "Missing tuning.metadata_episodic section."
        )

    alignment_result = tune_aux_alignment(
        source_single=source_single,
        base_config=aux_config,
        tuning_section=alignment_section,
        episodic_section=episodic_section,
        bundle=bundle,
        protocol=protocol,
        experiment_seed=experiment_seed,
        output_root=output_root,
        device=device,
    )

    aux_config = apply_overrides(
        config=aux_config,
        overrides=(
            alignment_result.best_overrides
        ),
    )

    # ========================================================
    # 4. BUILD FINAL TUNING DOUBLE ONCE
    #
    # Main is NOT retrained.
    # Only selected Aux configuration is trained once.
    # ========================================================

    print()
    print(
        "[TUNING] building final Double Head "
        "from the already-trained fixed source Single"
    )

    source_double = (
        _train_aux_from_fixed_single(
            source_single=source_single,
            config=aux_config,
            bundle=bundle,
            protocol=protocol,
            experiment_seed=experiment_seed,
            device=device,
            log_prefix=(
                "TTA SOURCE AUX FINAL"
            ),
        )
    )

    source_single.eval()
    source_double.eval()

    # ========================================================
    # SOURCE GRADIENT-NORM CALIBRATION
    #
    # Calibrate once on pseudo-source VALIDATION only.
    # No pseudo-OOD sample is used to construct thresholds.
    # ========================================================

    requires_gradient_norm_calibration = False

    for calibration_method_name in (
        "metadata_episodic",
        "metadata_cumulative",
    ):

        calibration_section = tuning_root.get(
            calibration_method_name
        )

        if (
            isinstance(
                calibration_section,
                Mapping,
            )
            and calibration_section.get(
                "gradient_norm_quantile_values"
            )
            is not None
        ):
            requires_gradient_norm_calibration = True
            break

    source_gradient_norms: (
        np.ndarray | None
    ) = None

    if requires_gradient_norm_calibration:

        source_slices = (
            protocol.pseudo_source_supervised()
        )

        (
            _,
            _,
            _,
            X_source_validation,
            _,
            y_aux_source_validation,
        ) = _concatenate_aux_source(
            source_slices
        )

        source_gradient_norms = (
            compute_source_aux_gradient_norms(
                model=source_double,
                X=X_source_validation,
                y_aux=y_aux_source_validation,
                device=device,
            )
        )

        print()
        print(
            "[TUNING][GRAD-CAL] "
            f"n={len(source_gradient_norms)} | "
            f"p25={np.quantile(source_gradient_norms, 0.25):.6f} | "
            f"p50={np.quantile(source_gradient_norms, 0.50):.6f} | "
            f"p75={np.quantile(source_gradient_norms, 0.75):.6f} | "
            f"p90={np.quantile(source_gradient_norms, 0.90):.6f}"
        )

    # ========================================================
    # 5. TTA
    # ========================================================

    method_results = {}

    for method_name in (
        "metadata_episodic",
        "metadata_cumulative",
    ):

        if method_name not in enabled_methods:
            continue

        section = tuning_root.get(
            method_name
        )

        if not isinstance(
            section,
            Mapping,
        ):
            raise ValueError(
                f"Missing tuning."
                f"{method_name}."
            )

        method_results[
            method_name
        ] = tune_tta_method(
            method_name=method_name,
            base_config=aux_config,
            tuning_section=section,
            inherited_overrides={},
            source_single=source_single,
            source_double=source_double,
            protocol=protocol,
            experiment_seed=experiment_seed,
            output_root=output_root,
            device=device,
            source_gradient_norms=(
                source_gradient_norms
            ),
        )

    cumulative = method_results.get(
        "metadata_cumulative"
    )

    # --------------------------------------------------------
    # Annual reset
    # --------------------------------------------------------

    method_name = (
        "metadata_cumulative_annual_reset"
    )

    if method_name in enabled_methods:

        if cumulative is None:
            raise RuntimeError(
                f"{method_name} requires "
                "metadata_cumulative tuning."
            )

        section = tuning_root.get(
            method_name,
            {
                "candidates": [
                    {
                        "id":
                            "inherit",

                        "parameters":
                            {},

                        "overrides":
                            {},
                    }
                ]
            },
        )

        method_results[
            method_name
        ] = tune_tta_method(
            method_name=method_name,
            base_config=aux_config,
            tuning_section=section,
            inherited_overrides=(
                _inherit_metadata_core(
                    cumulative,
                    target_method=method_name,
                )
            ),
            source_single=source_single,
            source_double=source_double,
            protocol=protocol,
            experiment_seed=experiment_seed,
            output_root=output_root,
            device=device,
        )

    # --------------------------------------------------------
    # Drift reset
    # --------------------------------------------------------

    drift_name = (
        "metadata_cumulative_drift_reset"
    )

    if drift_name in enabled_methods:

        if cumulative is None:
            raise RuntimeError(
                f"{drift_name} requires "
                "metadata_cumulative tuning."
            )

        section = tuning_root.get(
            drift_name
        )

        if not isinstance(
            section,
            Mapping,
        ):
            raise ValueError(
                f"Missing tuning."
                f"{drift_name}."
            )

        method_results[
            drift_name
        ] = tune_tta_method(
            method_name=drift_name,
            base_config=aux_config,
            tuning_section=section,
            inherited_overrides=(
                _inherit_metadata_core(
                    cumulative,
                    target_method=drift_name,
                )
            ),
            source_single=source_single,
            source_double=source_double,
            protocol=protocol,
            experiment_seed=experiment_seed,
            output_root=output_root,
            device=device,
        )

    drift_result = method_results.get(
        drift_name
    )

    # --------------------------------------------------------
    # Drift + annual
    # --------------------------------------------------------

    combined_name = (
        "metadata_cumulative_drift_annual_reset"
    )

    if combined_name in enabled_methods:

        if drift_result is None:
            raise RuntimeError(
                f"{combined_name} requires "
                f"{drift_name} tuning."
            )

        section = tuning_root.get(
            combined_name,
            {
                "candidates": [
                    {
                        "id":
                            "inherit",

                        "parameters":
                            {},

                        "overrides":
                            {},
                    }
                ]
            },
        )

        method_results[
            combined_name
        ] = tune_tta_method(
            method_name=combined_name,
            base_config=aux_config,
            tuning_section=section,
            inherited_overrides=(
                _inherit_drift_configuration(
                    cumulative,
                    drift_result,
                    target_method=combined_name,
                )
            ),
            source_single=source_single,
            source_double=source_double,
            protocol=protocol,
            experiment_seed=experiment_seed,
            output_root=output_root,
            device=device,
        )

    # --------------------------------------------------------
    # Temporal Gradient EMA
    # --------------------------------------------------------

    temporal_name = (
        "temporal_gradient_ema"
    )

    if temporal_name in enabled_methods:

        if cumulative is None:
            raise RuntimeError(
                f"{temporal_name} requires "
                "metadata_cumulative tuning."
            )

        section = tuning_root.get(
            temporal_name
        )

        if not isinstance(
            section,
            Mapping,
        ):
            raise ValueError(
                f"Missing tuning."
                f"{temporal_name}."
            )

        method_results[
            temporal_name
        ] = tune_tta_method(
            method_name=temporal_name,
            base_config=aux_config,
            tuning_section=section,
            inherited_overrides=(
                _inherit_metadata_core(
                    cumulative,
                    target_method=temporal_name,
                    include_gradient_norm_gate=False,
                )
            ),
            source_single=source_single,
            source_double=source_double,
            protocol=protocol,
            experiment_seed=experiment_seed,
            output_root=output_root,
            device=device,
        )

    # --------------------------------------------------------
    # TENT
    # --------------------------------------------------------

    if "tent" in enabled_methods:

        section = tuning_root.get(
            "tent"
        )

        if not isinstance(
            section,
            Mapping,
        ):
            raise ValueError(
                "Missing tuning.tent."
            )

        method_results[
            "tent"
        ] = tune_tta_method(
            method_name="tent",
            base_config=aux_config,
            tuning_section=section,
            inherited_overrides={},
            source_single=source_single,
            source_double=source_double,
            protocol=protocol,
            experiment_seed=experiment_seed,
            output_root=output_root,
            device=device,
        )

    if "cotta" in enabled_methods:
        raise NotImplementedError(
            "CoTTA is configured as a V3 placeholder "
            "but is not implemented yet."
        )

    return V3TuningResult(
        model=model_result,
        aux_head=aux_result,
        aux_alignment=alignment_result,
        methods=method_results,

        model_config=model_config,
        aux_config=aux_config,

        source_single=source_single,
        source_double=source_double,
    )