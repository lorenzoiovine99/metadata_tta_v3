from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
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
    ExperimentConfig,
)
from metadata_tta.data import (
    build_data_bundle,
)
from metadata_tta.protocols import (
    V3Protocol,
)
from metadata_tta.reproducibility import (
    set_seed,
)
from metadata_tta.tta import (
    get_method_class,
)
from metadata_tta.tuning.v3 import (
    train_tuning_source_models,
)


# ============================================================
# CLI
# ============================================================


def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Diagnose metadata/main gradient alignment on the "
            "HuffPost pseudo-OOD validation year."
        )
    )

    parser.add_argument(
        "--run-directory",
        required=True,
        help=(
            "Existing V4 HuffPost run directory containing "
            "run_config.yaml."
        ),
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "mps",
            "cuda",
        ],
        default="auto",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help=(
            "Optional limit for debugging. "
            "0 means use the complete pseudo-OOD validation stream."
        ),
    )

    return parser.parse_args()


# ============================================================
# DEVICE
# ============================================================


def _device(
    requested: str,
) -> torch.device:

    if requested == "cpu":
        return torch.device(
            "cpu"
        )

    if requested == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        return torch.device(
            "cuda"
        )

    if requested == "mps":

        if not (
            hasattr(
                torch.backends,
                "mps",
            )
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError(
                "MPS requested but unavailable."
            )

        return torch.device(
            "mps"
        )

    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    if (
        hasattr(
            torch.backends,
            "mps",
        )
        and torch.backends.mps.is_available()
    ):
        return torch.device(
            "mps"
        )

    return torch.device(
        "cpu"
    )


# ============================================================
# CONFIG
# ============================================================


def _load_run_config(
    run_directory: Path,
) -> ExperimentConfig:

    path = (
        run_directory
        / "run_config.yaml"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        payload = yaml.safe_load(
            handle
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "run_config.yaml must contain a mapping."
        )

    return ExperimentConfig(
        payload
    )


# ============================================================
# SOURCE REFERENCE
# ============================================================


def _source_reference(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    epsilon: float,
) -> dict[str, Any]:

    model.eval()

    with torch.no_grad():

        logits = model(
            x
        )

        if (
            logits.ndim != 2
            or logits.shape[0] != 1
            or logits.shape[1] < 2
        ):
            raise RuntimeError(
                "Expected main logits with shape [1, C], C >= 2."
            )

        probabilities = F.softmax(
            logits,
            dim=1,
        )

        entropy = -torch.sum(
            probabilities
            * torch.log(
                probabilities.clamp_min(
                    epsilon
                )
            ),
            dim=1,
        )

        n_classes = int(
            logits.shape[1]
        )

        confidence = (
            1.0
            - entropy
            / math.log(
                max(
                    n_classes,
                    2,
                )
            )
        ).clamp(
            min=0.0,
            max=1.0,
        )

        top_values, top_indices = (
            torch.topk(
                logits,
                k=2,
                dim=1,
            )
        )

        return {
            "top1":
                int(
                    top_indices[
                        0,
                        0,
                    ].item()
                ),

            "top2":
                int(
                    top_indices[
                        0,
                        1,
                    ].item()
                ),

            "confidence":
                float(
                    confidence[
                        0
                    ].item()
                ),

            "margin":
                float(
                    (
                        top_values[
                            0,
                            0,
                        ]
                        - top_values[
                            0,
                            1,
                        ]
                    ).item()
                ),
        }


# ============================================================
# GRADIENT HELPERS
# ============================================================


def _gradient_dot(
    first: list[torch.Tensor],
    second: list[torch.Tensor],
) -> float:

    if len(first) != len(second):
        raise RuntimeError(
            "Gradient-list size mismatch."
        )

    value = 0.0

    for first_gradient, second_gradient in zip(
        first,
        second,
    ):

        value += float(
            torch.sum(
                first_gradient
                .detach()
                .float()
                * second_gradient
                .detach()
                .float()
            ).item()
        )

    return float(
        value
    )


def _gradient_norm(
    gradients: list[torch.Tensor],
) -> float:

    squared = 0.0

    for gradient in gradients:

        squared += float(
            torch.sum(
                gradient
                .detach()
                .float()
                ** 2
            ).item()
        )

    return float(
        math.sqrt(
            max(
                squared,
                0.0,
            )
        )
    )


def _gradient_cosine(
    first: list[torch.Tensor],
    second: list[torch.Tensor],
    *,
    epsilon: float,
) -> float:

    first_norm = (
        _gradient_norm(
            first
        )
    )

    second_norm = (
        _gradient_norm(
            second
        )
    )

    denominator = (
        first_norm
        * second_norm
    )

    if denominator <= epsilon:
        return 0.0

    value = (
        _gradient_dot(
            first,
            second,
        )
        / denominator
    )

    return float(
        max(
            -1.0,
            min(
                1.0,
                value,
            ),
        )
    )


def _combine_gradients(
    first: list[torch.Tensor],
    second: list[torch.Tensor],
    second_scale: float,
) -> list[torch.Tensor]:

    if len(first) != len(second):
        raise RuntimeError(
            "Gradient-list size mismatch."
        )

    return [
        (
            first_gradient
            + float(
                second_scale
            )
            * second_gradient
        )
        for (
            first_gradient,
            second_gradient,
        )
        in zip(
            first,
            second,
        )
    ]


def _parameter_delta(
    *,
    parameters: list[torch.nn.Parameter],
    before: list[torch.Tensor],
) -> list[torch.Tensor]:

    return [
        (
            parameter.detach()
            - previous
        )
        for parameter, previous
        in zip(
            parameters,
            before,
        )
    ]


# ============================================================
# CURRENT MODEL MEASUREMENTS
# ============================================================


def _main_measurements(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    y_main: torch.Tensor,
    source_top1: int,
    source_top2: int,
) -> dict[str, Any]:

    model.eval()

    with torch.no_grad():

        logits = model(
            x
        )

        loss = F.cross_entropy(
            logits,
            y_main,
        )

        prediction = int(
            logits.argmax(
                dim=1
            )[0].item()
        )

        margin = float(
            (
                logits[
                    0,
                    source_top1,
                ]
                - logits[
                    0,
                    source_top2,
                ]
            ).item()
        )

    return {
        "loss":
            float(
                loss.item()
            ),

        "prediction":
            prediction,

        "correct":
            bool(
                prediction
                == int(
                    y_main[
                        0
                    ].item()
                )
            ),

        "source_selected_margin":
            margin,
    }


# ============================================================
# BOOLEAN METRICS
# ============================================================


def _safe_rate(
    rows: list[dict[str, Any]],
    key: str,
) -> float | None:

    values = [
        bool(
            row[
                key
            ]
        )
        for row in rows
        if row.get(
            key
        ) is not None
    ]

    if not values:
        return None

    return float(
        sum(
            values
        )
        / len(
            values
        )
    )


def _safe_mean(
    rows: list[dict[str, Any]],
    key: str,
) -> float | None:

    values = []

    for row in rows:

        value = row.get(
            key
        )

        if value is None:
            continue

        value = float(
            value
        )

        if not math.isfinite(
            value
        ):
            continue

        values.append(
            value
        )

    if not values:
        return None

    return float(
        np.mean(
            values
        )
    )


def _confusion_metrics(
    rows: list[dict[str, Any]],
    *,
    predicted_key: str,
    target_key: str,
) -> dict[str, Any]:

    selected = [
        row
        for row in rows
        if (
            row.get(
                predicted_key
            )
            is not None
            and row.get(
                target_key
            )
            is not None
        )
    ]

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for row in selected:

        predicted = bool(
            row[
                predicted_key
            ]
        )

        target = bool(
            row[
                target_key
            ]
        )

        if predicted and target:
            tp += 1

        elif predicted and not target:
            fp += 1

        elif (
            not predicted
            and target
        ):
            fn += 1

        else:
            tn += 1

    precision = (
        float(
            tp
            / (
                tp
                + fp
            )
        )
        if (
            tp
            + fp
        ) > 0
        else None
    )

    recall = (
        float(
            tp
            / (
                tp
                + fn
            )
        )
        if (
            tp
            + fn
        ) > 0
        else None
    )

    specificity = (
        float(
            tn
            / (
                tn
                + fp
            )
        )
        if (
            tn
            + fp
        ) > 0
        else None
    )

    return {
        "n":
            len(
                selected
            ),

        "tp":
            tp,

        "fp":
            fp,

        "tn":
            tn,

        "fn":
            fn,

        "precision":
            precision,

        "recall":
            recall,

        "specificity":
            specificity,
    }


# ============================================================
# BINNED STATS
# ============================================================


def _binned_statistics(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    edges: list[float],
) -> list[dict[str, Any]]:

    output = []

    for index in range(
        len(
            edges
        )
        - 1
    ):

        lower = float(
            edges[
                index
            ]
        )

        upper = float(
            edges[
                index
                + 1
            ]
        )

        is_last = (
            index
            == len(
                edges
            )
            - 2
        )

        selected = []

        for row in rows:

            raw_value = row.get(
                value_key
            )

            if raw_value is None:
                continue

            value = float(
                raw_value
            )

            if is_last:

                inside = (
                    lower
                    <= value
                    <= upper
                )

            else:

                inside = (
                    lower
                    <= value
                    < upper
                )

            if inside:
                selected.append(
                    row
                )

        output.append(
            {
                "lower":
                    lower,

                "upper":
                    upper,

                "n":
                    len(
                        selected
                    ),

                "source_accuracy":
                    _safe_rate(
                        selected,
                        "source_correct",
                    ),

                "raw_true_conflict_rate":
                    _safe_rate(
                        selected,
                        "raw_true_conflict",
                    ),

                "guard_predicted_conflict_rate":
                    _safe_rate(
                        selected,
                        "guard_predicted_conflict",
                    ),

                "actual_main_loss_worsened_rate":
                    _safe_rate(
                        selected,
                        "actual_main_loss_worsened",
                    ),

                "actual_update_harmful_rate":
                    _safe_rate(
                        selected,
                        "actual_update_harmful_first_order",
                    ),

                "mean_aux_main_cosine":
                    _safe_mean(
                        selected,
                        "aux_main_cosine",
                    ),

                "mean_aux_guard_cosine":
                    _safe_mean(
                        selected,
                        "aux_guard_cosine",
                    ),
            }
        )

    return output


# ============================================================
# SUMMARY
# ============================================================


def _trajectory_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:

    eligible = [
        row
        for row in rows
        if bool(
            row[
                "gate_pass"
            ]
        )
    ]

    source_correct = [
        row
        for row in eligible
        if bool(
            row[
                "source_correct"
            ]
        )
    ]

    source_wrong = [
        row
        for row in eligible
        if not bool(
            row[
                "source_correct"
            ]
        )
    ]

    correct_to_wrong = sum(
        bool(
            row[
                "current_correct_before"
            ]
        )
        and not bool(
            row[
                "current_correct_after"
            ]
        )
        for row in eligible
    )

    wrong_to_correct = sum(
        not bool(
            row[
                "current_correct_before"
            ]
        )
        and bool(
            row[
                "current_correct_after"
            ]
        )
        for row in eligible
    )

    return {
        "n_samples":
            len(
                rows
            ),

        "n_gate_pass":
            len(
                eligible
            ),

        "gate_pass_rate":
            float(
                len(
                    eligible
                )
                / max(
                    len(
                        rows
                    ),
                    1,
                )
            ),

        "source_accuracy_gate_pass":
            _safe_rate(
                eligible,
                "source_correct",
            ),

        "current_accuracy_before":
            _safe_rate(
                eligible,
                "current_correct_before",
            ),

        "current_accuracy_after":
            _safe_rate(
                eligible,
                "current_correct_after",
            ),

        "raw_true_conflict_rate":
            _safe_rate(
                eligible,
                "raw_true_conflict",
            ),

        "guard_predicted_conflict_rate":
            _safe_rate(
                eligible,
                "guard_predicted_conflict",
            ),

        "mean_aux_main_cosine":
            _safe_mean(
                eligible,
                "aux_main_cosine",
            ),

        "mean_aux_guard_cosine":
            _safe_mean(
                eligible,
                "aux_guard_cosine",
            ),

        "mean_safe_main_cosine":
            _safe_mean(
                eligible,
                "safe_main_cosine",
            ),

        "mean_total_main_cosine":
            _safe_mean(
                eligible,
                "total_main_cosine",
            ),

        "actual_update_harmful_rate":
            _safe_rate(
                eligible,
                "actual_update_harmful_first_order",
            ),

        "actual_margin_harmful_rate":
            _safe_rate(
                eligible,
                "actual_margin_harmful_first_order",
            ),

        "actual_main_loss_worsened_rate":
            _safe_rate(
                eligible,
                "actual_main_loss_worsened",
            ),

        "actual_source_margin_decreased_rate":
            _safe_rate(
                eligible,
                "actual_source_margin_decreased",
            ),

        "mean_main_loss_delta":
            _safe_mean(
                eligible,
                "main_loss_delta",
            ),

        "mean_source_margin_delta":
            _safe_mean(
                eligible,
                "source_margin_delta",
            ),

        "mean_source_confidence":
            _safe_mean(
                eligible,
                "source_confidence",
            ),

        "mean_normalized_aux_loss":
            _safe_mean(
                eligible,
                "normalized_aux_loss",
            ),

        "mean_raw_aux_gradient_norm":
            _safe_mean(
                eligible,
                "raw_aux_gradient_norm",
            ),

        "mean_safe_aux_gradient_norm":
            _safe_mean(
                eligible,
                "safe_aux_gradient_norm",
            ),

        "mean_parameter_update_norm":
            _safe_mean(
                eligible,
                "parameter_update_norm",
            ),

        "mean_removed_gradient_fraction":
            _safe_mean(
                eligible,
                "removed_gradient_fraction",
            ),

        "mean_projection_strength":
            _safe_mean(
                eligible,
                "projection_strength",
            ),

        "prediction_flips":
            {
                "correct_to_wrong":
                    int(
                        correct_to_wrong
                    ),

                "wrong_to_correct":
                    int(
                        wrong_to_correct
                    ),
            },

        "guard_vs_raw_true_conflict":
            _confusion_metrics(
                eligible,
                predicted_key=(
                    "guard_predicted_conflict"
                ),
                target_key=(
                    "raw_true_conflict"
                ),
            ),

        "guard_vs_actual_update_harm":
            _confusion_metrics(
                eligible,
                predicted_key=(
                    "guard_predicted_conflict"
                ),
                target_key=(
                    "actual_update_harmful_first_order"
                ),
            ),

        "guard_vs_actual_main_loss_worsened":
            _confusion_metrics(
                eligible,
                predicted_key=(
                    "guard_predicted_conflict"
                ),
                target_key=(
                    "actual_main_loss_worsened"
                ),
            ),

        "source_correct_subset":
            {
                "n":
                    len(
                        source_correct
                    ),

                "raw_true_conflict_rate":
                    _safe_rate(
                        source_correct,
                        "raw_true_conflict",
                    ),

                "guard_predicted_conflict_rate":
                    _safe_rate(
                        source_correct,
                        "guard_predicted_conflict",
                    ),

                "actual_main_loss_worsened_rate":
                    _safe_rate(
                        source_correct,
                        "actual_main_loss_worsened",
                    ),

                "mean_aux_main_cosine":
                    _safe_mean(
                        source_correct,
                        "aux_main_cosine",
                    ),
            },

        "source_wrong_subset":
            {
                "n":
                    len(
                        source_wrong
                    ),

                "raw_true_conflict_rate":
                    _safe_rate(
                        source_wrong,
                        "raw_true_conflict",
                    ),

                "guard_predicted_conflict_rate":
                    _safe_rate(
                        source_wrong,
                        "guard_predicted_conflict",
                    ),

                "actual_main_loss_worsened_rate":
                    _safe_rate(
                        source_wrong,
                        "actual_main_loss_worsened",
                    ),

                "mean_aux_main_cosine":
                    _safe_mean(
                        source_wrong,
                        "aux_main_cosine",
                    ),
            },

        "confidence_bins":
            _binned_statistics(
                eligible,
                value_key=(
                    "source_confidence"
                ),
                edges=[
                    0.0,
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                    1.0,
                ],
            ),

        "aux_loss_bins":
            _binned_statistics(
                eligible,
                value_key=(
                    "normalized_aux_loss"
                ),
                edges=[
                    0.0,
                    0.25,
                    0.5,
                    0.75,
                    1.0,
                    2.0,
                ],
            ),
    }


# ============================================================
# CSV
# ============================================================


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:

    if not rows:
        raise RuntimeError(
            "No diagnostic rows generated."
        )

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row
        }
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# TRAJECTORY
# ============================================================


def _run_trajectory(
    *,
    trajectory_name: str,
    source_model: torch.nn.Module,
    method_config: dict[str, Any],
    X: np.ndarray,
    y_main: np.ndarray,
    y_aux: np.ndarray,
    device: torch.device,
    experiment_seed: int,
    max_samples: int,
) -> list[dict[str, Any]]:

    if trajectory_name not in {
        "v3",
        "v4",
    }:
        raise ValueError(
            trajectory_name
        )

    config = copy.deepcopy(
        method_config
    )

    config[
        "main_preservation_enabled"
    ] = (
        trajectory_name
        == "v4"
    )

    method_class = (
        get_method_class(
            "metadata_cumulative"
        )
    )

    set_seed(
        experiment_seed
    )

    method = method_class(
        source_model=copy.deepcopy(
            source_model
        ),
        config=config,
        device=device,
    )

    if int(
        method.steps
    ) != 1:
        raise RuntimeError(
            "Diagnostic currently requires "
            "methods.metadata_cumulative.steps == 1."
        )

    pristine_source = copy.deepcopy(
        source_model
    ).to(
        device
    )

    pristine_source.eval()

    n_samples = len(
        X
    )

    if max_samples > 0:

        n_samples = min(
            n_samples,
            max_samples,
        )

    rows: list[
        dict[str, Any]
    ] = []

    print()
    print("=" * 80)
    print(
        f"DIAGNOSTIC TRAJECTORY: "
        f"{trajectory_name.upper()}"
    )
    print("=" * 80)
    print(
        "main_preservation_enabled="
        f"{config['main_preservation_enabled']}"
    )
    print(
        "window="
        f"[{method.normalized_aux_loss_min}, "
        f"{method.normalized_aux_loss_max}]"
    )
    print(
        f"learning_rate="
        f"{method.learning_rate}"
    )
    print(
        f"regularization="
        f"{method.regularization}"
    )
    print(
        f"samples={n_samples}"
    )

    for index in range(
        n_samples
    ):

        x = torch.as_tensor(
            X[
                index:index + 1
            ],
            dtype=torch.float32,
            device=device,
        )

        y_main_tensor = torch.as_tensor(
            [
                int(
                    y_main[
                        index
                    ]
                )
            ],
            dtype=torch.long,
            device=device,
        )

        y_aux_tensor = torch.as_tensor(
            [
                int(
                    y_aux[
                        index
                    ]
                )
            ],
            dtype=torch.long,
            device=device,
        )

        # ----------------------------------------------------
        # Immutable source prediction
        # ----------------------------------------------------

        source = _source_reference(
            model=pristine_source,
            x=x,
            epsilon=method.epsilon,
        )

        source_correct = bool(
            source[
                "top1"
            ]
            == int(
                y_main_tensor[
                    0
                ].item()
            )
        )

        # ----------------------------------------------------
        # Gate loss in CURRENT trajectory state
        # ----------------------------------------------------

        method.model.eval()

        with torch.no_grad():

            gate_aux_loss = (
                method._auxiliary_loss(
                    x_tensor=x,
                    y_aux_tensor=(
                        y_aux_tensor
                    ),
                )
            )

        aux_loss_value = float(
            gate_aux_loss.item()
        )

        normalized_aux_loss = float(
            aux_loss_value
            / max(
                method.aux_loss_normalizer,
                method.epsilon,
            )
        )

        gate_pass = bool(
            normalized_aux_loss
            >= method.normalized_aux_loss_min
            and normalized_aux_loss
            <= method.normalized_aux_loss_max
        )

        current_before = (
            _main_measurements(
                model=method.model,
                x=x,
                y_main=y_main_tensor,
                source_top1=(
                    source[
                        "top1"
                    ]
                ),
                source_top2=(
                    source[
                        "top2"
                    ]
                ),
            )
        )

        row: dict[
            str,
            Any,
        ] = {
            "trajectory":
                trajectory_name,

            "sample_index":
                index,

            "main_label":
                int(
                    y_main_tensor[
                        0
                    ].item()
                ),

            "aux_label":
                int(
                    y_aux_tensor[
                        0
                    ].item()
                ),

            "source_top1":
                int(
                    source[
                        "top1"
                    ]
                ),

            "source_top2":
                int(
                    source[
                        "top2"
                    ]
                ),

            "source_correct":
                source_correct,

            "source_confidence":
                float(
                    source[
                        "confidence"
                    ]
                ),

            "source_margin":
                float(
                    source[
                        "margin"
                    ]
                ),

            "aux_loss":
                aux_loss_value,

            "normalized_aux_loss":
                normalized_aux_loss,

            "gate_pass":
                gate_pass,

            "current_prediction_before":
                int(
                    current_before[
                        "prediction"
                    ]
                ),

            "current_correct_before":
                bool(
                    current_before[
                        "correct"
                    ]
                ),

            "main_loss_before":
                float(
                    current_before[
                        "loss"
                    ]
                ),

            "source_selected_margin_before":
                float(
                    current_before[
                        "source_selected_margin"
                    ]
                ),
        }

        # ----------------------------------------------------
        # Gate failed: observe anyway to guarantee exact stream
        # semantics, then continue.
        # ----------------------------------------------------

        if not gate_pass:

            result = method.observe(
                x=x,
                y_aux=y_aux_tensor,
            )

            if result.applied:
                raise RuntimeError(
                    "Diagnostic gate disagrees "
                    "with MetadataTTA.observe()."
                )

            row.update(
                {
                    "guard_predicted_conflict":
                        None,

                    "raw_true_conflict":
                        None,

                    "actual_update_harmful_first_order":
                        None,

                    "actual_margin_harmful_first_order":
                        None,

                    "actual_main_loss_worsened":
                        None,

                    "actual_source_margin_decreased":
                        None,
                }
            )

            rows.append(
                row
            )

            continue

        # ----------------------------------------------------
        # RAW AUX GRADIENT
        # ----------------------------------------------------

        aux_loss = (
            method._auxiliary_loss(
                x_tensor=x,
                y_aux_tensor=(
                    y_aux_tensor
                ),
            )
        )

        raw_aux_gradients = (
            method._autograd_gradient_list(
                aux_loss
            )
        )

        # ----------------------------------------------------
        # TRUE MAIN GRADIENT
        #
        # DIAGNOSTIC ONLY.
        # y_main is NEVER passed to method.observe().
        # ----------------------------------------------------

        main_logits = method.model(
            x
        )

        true_main_loss = (
            F.cross_entropy(
                main_logits,
                y_main_tensor,
            )
        )

        true_main_gradients = (
            method._autograd_gradient_list(
                true_main_loss
            )
        )

        # ----------------------------------------------------
        # V4 GUARD GRADIENT
        # ----------------------------------------------------

        top1_tensor = torch.as_tensor(
            [
                int(
                    source[
                        "top1"
                    ]
                )
            ],
            dtype=torch.long,
            device=device,
        )

        top2_tensor = torch.as_tensor(
            [
                int(
                    source[
                        "top2"
                    ]
                )
            ],
            dtype=torch.long,
            device=device,
        )

        guard_gradients = (
            method._main_guard_gradient(
                x_tensor=x,
                top1_indices=(
                    top1_tensor
                ),
                top2_indices=(
                    top2_tensor
                ),
            )
        )

        # ----------------------------------------------------
        # SOURCE ANCHOR GRADIENT
        # ----------------------------------------------------

        anchor_loss = (
            method._source_anchor_loss()
        )

        anchor_gradients = (
            method._autograd_gradient_list(
                anchor_loss
            )
        )

        # ----------------------------------------------------
        # WHAT V4 WOULD PROJECT
        # ----------------------------------------------------

        (
            projected_aux_gradients,
            surgery_diagnostics,
        ) = (
            method
            ._main_preserving_aux_gradient(
                raw_aux_gradients=(
                    raw_aux_gradients
                ),
                guard_gradients=(
                    guard_gradients
                ),
                source_confidence=float(
                    source[
                        "confidence"
                    ]
                ),
            )
        )

        if trajectory_name == "v4":

            accepted_aux_gradients = (
                projected_aux_gradients
            )

        else:

            accepted_aux_gradients = (
                raw_aux_gradients
            )

        total_gradients = (
            _combine_gradients(
                accepted_aux_gradients,
                anchor_gradients,
                method.regularization,
            )
        )

        # ----------------------------------------------------
        # RAW GRADIENT GEOMETRY
        # ----------------------------------------------------

        aux_main_dot = (
            _gradient_dot(
                raw_aux_gradients,
                true_main_gradients,
            )
        )

        aux_guard_dot = (
            _gradient_dot(
                raw_aux_gradients,
                guard_gradients,
            )
        )

        safe_main_dot = (
            _gradient_dot(
                accepted_aux_gradients,
                true_main_gradients,
            )
        )

        total_main_dot = (
            _gradient_dot(
                total_gradients,
                true_main_gradients,
            )
        )

        raw_true_conflict = bool(
            aux_main_dot
            < 0.0
        )

        guard_predicted_conflict = bool(
            aux_guard_dot
            < 0.0
        )

        # ----------------------------------------------------
        # SAVE PARAMETER STATE, THEN EXECUTE REAL METHOD UPDATE
        # ----------------------------------------------------

        parameters_before = [
            parameter
            .detach()
            .clone()
            for parameter
            in method.trainable_parameters
        ]

        result = method.observe(
            x=x,
            y_aux=y_aux_tensor,
        )

        if not result.applied:
            raise RuntimeError(
                "Diagnostic expected an update "
                "but MetadataTTA.observe() rejected it."
            )

        actual_parameter_delta = (
            _parameter_delta(
                parameters=(
                    method.trainable_parameters
                ),
                before=(
                    parameters_before
                ),
            )
        )

        # ----------------------------------------------------
        # ACTUAL UPDATE GEOMETRY AFTER CLIPPING + ADAM
        #
        # main gradient:
        #   g_main dot delta_theta > 0
        # means first-order MAIN LOSS INCREASE.
        #
        # guard gradient = grad(-margin):
        #   g_guard dot delta_theta > 0
        # means first-order SOURCE MARGIN DECREASE.
        # ----------------------------------------------------

        main_update_dot = (
            _gradient_dot(
                true_main_gradients,
                actual_parameter_delta,
            )
        )

        guard_update_dot = (
            _gradient_dot(
                guard_gradients,
                actual_parameter_delta,
            )
        )

        actual_update_harmful = bool(
            main_update_dot
            > 0.0
        )

        actual_margin_harmful = bool(
            guard_update_dot
            > 0.0
        )

        # ----------------------------------------------------
        # EXACT POST-UPDATE MEASUREMENTS
        # ----------------------------------------------------

        current_after = (
            _main_measurements(
                model=method.model,
                x=x,
                y_main=y_main_tensor,
                source_top1=(
                    source[
                        "top1"
                    ]
                ),
                source_top2=(
                    source[
                        "top2"
                    ]
                ),
            )
        )

        main_loss_delta = float(
            current_after[
                "loss"
            ]
            - current_before[
                "loss"
            ]
        )

        source_margin_delta = float(
            current_after[
                "source_selected_margin"
            ]
            - current_before[
                "source_selected_margin"
            ]
        )

        # ----------------------------------------------------
        # RECORD
        # ----------------------------------------------------

        row.update(
            {
                "raw_aux_gradient_norm":
                    _gradient_norm(
                        raw_aux_gradients
                    ),

                "safe_aux_gradient_norm":
                    _gradient_norm(
                        accepted_aux_gradients
                    ),

                "true_main_gradient_norm":
                    _gradient_norm(
                        true_main_gradients
                    ),

                "guard_gradient_norm":
                    _gradient_norm(
                        guard_gradients
                    ),

                "anchor_gradient_norm":
                    _gradient_norm(
                        anchor_gradients
                    ),

                "total_gradient_norm":
                    _gradient_norm(
                        total_gradients
                    ),

                "parameter_update_norm":
                    _gradient_norm(
                        actual_parameter_delta
                    ),

                "aux_main_dot":
                    aux_main_dot,

                "aux_guard_dot":
                    aux_guard_dot,

                "safe_main_dot":
                    safe_main_dot,

                "total_main_dot":
                    total_main_dot,

                "aux_main_cosine":
                    _gradient_cosine(
                        raw_aux_gradients,
                        true_main_gradients,
                        epsilon=method.epsilon,
                    ),

                "aux_guard_cosine":
                    _gradient_cosine(
                        raw_aux_gradients,
                        guard_gradients,
                        epsilon=method.epsilon,
                    ),

                "safe_main_cosine":
                    _gradient_cosine(
                        accepted_aux_gradients,
                        true_main_gradients,
                        epsilon=method.epsilon,
                    ),

                "total_main_cosine":
                    _gradient_cosine(
                        total_gradients,
                        true_main_gradients,
                        epsilon=method.epsilon,
                    ),

                "raw_true_conflict":
                    raw_true_conflict,

                "guard_predicted_conflict":
                    guard_predicted_conflict,

                "projection_strength":
                    float(
                        surgery_diagnostics[
                            "projection_strength"
                        ]
                    ),

                "removed_gradient_fraction":
                    float(
                        surgery_diagnostics[
                            "removed_gradient_fraction"
                        ]
                    ),

                "main_update_dot":
                    main_update_dot,

                "guard_update_dot":
                    guard_update_dot,

                "actual_update_harmful_first_order":
                    actual_update_harmful,

                "actual_margin_harmful_first_order":
                    actual_margin_harmful,

                "main_loss_after":
                    float(
                        current_after[
                            "loss"
                        ]
                    ),

                "main_loss_delta":
                    main_loss_delta,

                "actual_main_loss_worsened":
                    bool(
                        main_loss_delta
                        > 0.0
                    ),

                "source_selected_margin_after":
                    float(
                        current_after[
                            "source_selected_margin"
                        ]
                    ),

                "source_margin_delta":
                    source_margin_delta,

                "actual_source_margin_decreased":
                    bool(
                        source_margin_delta
                        < 0.0
                    ),

                "current_prediction_after":
                    int(
                        current_after[
                            "prediction"
                        ]
                    ),

                "current_correct_after":
                    bool(
                        current_after[
                            "correct"
                        ]
                    ),
            }
        )

        rows.append(
            row
        )

        if (
            (
                index
                + 1
            )
            % 250
            == 0
        ):
            print(
                f"[{trajectory_name}] "
                f"processed "
                f"{index + 1}/{n_samples}"
            )

    return rows


# ============================================================
# PRINT SUMMARY
# ============================================================


def _print_trajectory_summary(
    name: str,
    summary: dict[str, Any],
) -> None:

    print()
    print("=" * 80)
    print(
        f"{name.upper()} SUMMARY"
    )
    print("=" * 80)

    print(
        "samples:",
        summary[
            "n_samples"
        ],
    )

    print(
        "gate pass:",
        summary[
            "n_gate_pass"
        ],
        f"({summary['gate_pass_rate']:.4f})",
    )

    print(
        "source accuracy on gate-pass:",
        summary[
            "source_accuracy_gate_pass"
        ],
    )

    print(
        "raw TRUE aux/main conflict rate:",
        summary[
            "raw_true_conflict_rate"
        ],
    )

    print(
        "V4 guard predicted conflict rate:",
        summary[
            "guard_predicted_conflict_rate"
        ],
    )

    print(
        "mean cos(aux, true-main):",
        summary[
            "mean_aux_main_cosine"
        ],
    )

    print(
        "mean cos(aux, guard):",
        summary[
            "mean_aux_guard_cosine"
        ],
    )

    print(
        "actual harmful update rate "
        "(after optimizer, first-order):",
        summary[
            "actual_update_harmful_rate"
        ],
    )

    print(
        "actual main-loss worsened rate:",
        summary[
            "actual_main_loss_worsened_rate"
        ],
    )

    print(
        "actual source-margin decreased rate:",
        summary[
            "actual_source_margin_decreased_rate"
        ],
    )

    print(
        "mean main-loss delta:",
        summary[
            "mean_main_loss_delta"
        ],
    )

    print(
        "mean source-margin delta:",
        summary[
            "mean_source_margin_delta"
        ],
    )

    print()
    print(
        "Guard vs TRUE raw aux/main conflict:"
    )

    print(
        json.dumps(
            summary[
                "guard_vs_raw_true_conflict"
            ],
            indent=2,
        )
    )

    print()
    print(
        "Guard vs ACTUAL harmful optimizer update:"
    )

    print(
        json.dumps(
            summary[
                "guard_vs_actual_update_harm"
            ],
            indent=2,
        )
    )

    print()
    print(
        "Guard vs ACTUAL main-loss worsening:"
    )

    print(
        json.dumps(
            summary[
                "guard_vs_actual_main_loss_worsened"
            ],
            indent=2,
        )
    )

    print()
    print(
        "Source-correct subset:"
    )

    print(
        json.dumps(
            summary[
                "source_correct_subset"
            ],
            indent=2,
        )
    )

    print()
    print(
        "Source-wrong subset:"
    )

    print(
        json.dumps(
            summary[
                "source_wrong_subset"
            ],
            indent=2,
        )
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    args = _parse_args()

    run_directory = Path(
        args.run_directory
    ).expanduser().resolve()

    if not run_directory.exists():
        raise FileNotFoundError(
            run_directory
        )

    config = _load_run_config(
        run_directory
    )

    if (
        config.dataset_name
        != "huffpost"
    ):
        raise RuntimeError(
            "This diagnostic is intentionally restricted "
            "to HuffPost."
        )

    device = _device(
        args.device
    )

    print()
    print("=" * 80)
    print(
        "HUFFPOST METADATA-GRADIENT DIAGNOSTIC"
    )
    print("=" * 80)
    print(
        "run_directory=",
        run_directory,
    )
    print(
        "experiment_seed=",
        config.experiment_seed,
    )
    print(
        "split_seed=",
        config.split_seed,
    )
    print(
        "device=",
        device,
    )

    # ========================================================
    # DATA + PROTOCOL
    # ========================================================

    bundle = build_data_bundle(
        config
    )

    protocol = V3Protocol(
        bundle=bundle,
        dataset_config=(
            config.section(
                "dataset"
            )
        ),
        protocol_config=(
            config.section(
                "protocol"
            )
        ),
        split_seed=(
            config.split_seed
        ),
    )

    protocol.verify_split_disjointness()

    if tuple(
        protocol.years.pseudo_source
    ) != (
        2012,
        2013,
        2014,
    ):
        raise RuntimeError(
            "Expected HuffPost pseudo-source "
            "2012-2014, got "
            f"{protocol.years.pseudo_source}."
        )

    if tuple(
        protocol.years.pseudo_ood
    ) != (
        2015,
    ):
        raise RuntimeError(
            "Expected HuffPost pseudo-OOD 2015, got "
            f"{protocol.years.pseudo_ood}."
        )

    # ========================================================
    # RECREATE EXACT TUNING SOURCE MODEL
    #
    # run_config.yaml already contains the selected model and
    # Aux hyperparameters. Additional TTA hyperparameters in
    # the config do not affect source training.
    # ========================================================

    print()
    print("=" * 80)
    print(
        "RETRAINING PSEUDO-SOURCE 2012-2014"
    )
    print("=" * 80)

    (
        _source_single,
        source_double,
    ) = train_tuning_source_models(
        config=config,
        bundle=bundle,
        protocol=protocol,
        experiment_seed=(
            config.experiment_seed
        ),
        device=device,
    )

    source_double.eval()

    # ========================================================
    # PSEUDO-OOD VALIDATION ONLY
    # ========================================================

    pseudo_ood = (
        protocol
        .pseudo_ood_validation_stream()
    )

    if len(
        pseudo_ood
    ) != 1:
        raise RuntimeError(
            "Expected exactly one HuffPost "
            "pseudo-OOD validation year."
        )

    stream = pseudo_ood[
        0
    ]

    if int(
        stream.year
    ) != 2015:
        raise RuntimeError(
            f"Expected year 2015, got {stream.year}."
        )

    print()
    print(
        "Diagnostic stream:"
    )
    print(
        f"year={stream.year}"
    )
    print(
        f"split={stream.split_name}"
    )
    print(
        f"n={len(stream.X)}"
    )

    # ========================================================
    # SELECTED CUMULATIVE CONFIG
    # ========================================================

    cumulative_config = dict(
        config.method_config(
            "metadata_cumulative"
        )
    )

    print()
    print(
        "Selected cumulative config:"
    )

    for key in (
        "learning_rate",
        "regularization",
        "gradient_clip",
        "normalized_aux_loss_min",
        "normalized_aux_loss_max",
        "steps",
        "main_preservation_enabled",
    ):

        if key in cumulative_config:

            print(
                f"  {key}="
                f"{cumulative_config[key]}"
            )

    # ========================================================
    # V3 TRAJECTORY
    # ========================================================

    v3_rows = _run_trajectory(
        trajectory_name="v3",
        source_model=source_double,
        method_config=(
            cumulative_config
        ),
        X=stream.X,
        y_main=stream.y_main,
        y_aux=stream.y_aux,
        device=device,
        experiment_seed=(
            config.experiment_seed
        ),
        max_samples=int(
            args.max_samples
        ),
    )

    # ========================================================
    # V4 TRAJECTORY
    # ========================================================

    v4_rows = _run_trajectory(
        trajectory_name="v4",
        source_model=source_double,
        method_config=(
            cumulative_config
        ),
        X=stream.X,
        y_main=stream.y_main,
        y_aux=stream.y_aux,
        device=device,
        experiment_seed=(
            config.experiment_seed
        ),
        max_samples=int(
            args.max_samples
        ),
    )

    # ========================================================
    # SUMMARIZE
    # ========================================================

    v3_summary = (
        _trajectory_summary(
            v3_rows
        )
    )

    v4_summary = (
        _trajectory_summary(
            v4_rows
        )
    )

    output_directory = (
        run_directory
        / "analysis"
        / "huffpost_gradient_alignment"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_csv(
        output_directory
        / "v3_trajectory.csv",
        v3_rows,
    )

    _write_csv(
        output_directory
        / "v4_trajectory.csv",
        v4_rows,
    )

    summary = {
        "dataset":
            "huffpost",

        "experiment_seed":
            int(
                config.experiment_seed
            ),

        "split_seed":
            int(
                config.split_seed
            ),

        "pseudo_source_years":
            list(
                protocol.years.pseudo_source
            ),

        "pseudo_ood_years":
            list(
                protocol.years.pseudo_ood
            ),

        "diagnostic_split":
            "validation",

        "uses_main_labels_for_adaptation":
            False,

        "note":
            (
                "2015 main labels are used only for "
                "diagnostic gradient/alignment measurements. "
                "method.observe() receives only y_aux."
            ),

        "v3":
            v3_summary,

        "v4":
            v4_summary,
    }

    with (
        output_directory
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
            sort_keys=True,
        )

    _print_trajectory_summary(
        "v3",
        v3_summary,
    )

    _print_trajectory_summary(
        "v4",
        v4_summary,
    )

    print()
    print("=" * 80)
    print(
        "DIAGNOSTIC COMPLETE"
    )
    print("=" * 80)
    print(
        "output_directory=",
        output_directory,
    )
    print(
        "summary=",
        (
            output_directory
            / "summary.json"
        ),
    )


if __name__ == "__main__":
    main()