from __future__ import annotations

import argparse
import copy
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


from metadata_tta.config import ExperimentConfig
from metadata_tta.data import build_data_bundle
from metadata_tta.protocols import V3Protocol
from metadata_tta.reproducibility import set_seed
from metadata_tta.tta import get_method_class
from metadata_tta.tuning.v3 import (
    train_tuning_source_models,
)


# ============================================================
# CLI
# ============================================================


def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Oracle diagnostic for HuffPost metadata TTA."
        )
    )

    parser.add_argument(
        "--run-directory",
        required=True,
        help=(
            "Existing tune_and_test run directory containing "
            "the selected run_config.yaml."
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
            "0 = full 2015 pseudo-OOD validation stream."
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


def _load_config(
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

    denominator = (
        _gradient_norm(
            first
        )
        * _gradient_norm(
            second
        )
    )

    if denominator <= epsilon:
        return 0.0

    cosine = (
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
                cosine,
            ),
        )
    )


def _oracle_project(
    aux_gradients: list[torch.Tensor],
    main_gradients: list[torch.Tensor],
    *,
    epsilon: float,
) -> list[torch.Tensor]:
    """
    Full oracle projection.

    We require:

        g_aux dot g_main >= 0

    because the actual descent step is -g_aux and therefore

        delta L_main ~= -eta * g_main dot g_aux.

    If dot < 0, remove exactly the main-conflicting component.
    """

    dot_product = (
        _gradient_dot(
            aux_gradients,
            main_gradients,
        )
    )

    main_norm = (
        _gradient_norm(
            main_gradients
        )
    )

    if (
        dot_product >= 0.0
        or main_norm <= epsilon
    ):
        return [
            gradient.detach().clone()
            for gradient in aux_gradients
        ]

    coefficient = float(
        dot_product
        / (
            main_norm ** 2
            + epsilon
        )
    )

    return [
        (
            aux_gradient
            - coefficient
            * main_gradient
        )
        for aux_gradient, main_gradient
        in zip(
            aux_gradients,
            main_gradients,
        )
    ]


# ============================================================
# METRICS
# ============================================================


def _predict(
    model: torch.nn.Module,
    x: torch.Tensor,
) -> int:

    model.eval()

    with torch.no_grad():

        logits = model(
            x
        )

    return int(
        logits.argmax(
            dim=1
        )[0].item()
    )


def _main_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_main: torch.Tensor,
) -> float:

    model.eval()

    with torch.no_grad():

        logits = model(
            x
        )

        loss = F.cross_entropy(
            logits,
            y_main,
        )

    return float(
        loss.item()
    )


# ============================================================
# EXACT MANUAL UPDATE
# ============================================================


def _apply_gradient_update(
    *,
    method,
    aux_gradients: list[torch.Tensor],
) -> float:
    """
    Apply exactly the same optimizer mechanics as MetadataTTA:

      aux gradient
      + configured source-anchor gradient
      -> gradient clipping
      -> Adam step

    Returns adapter parameter delta norm.
    """

    anchor_gradients = None

    if method.regularization > 0.0:

        anchor_loss = (
            method._source_anchor_loss()
        )

        anchor_gradients = (
            method._autograd_gradient_list(
                anchor_loss
            )
        )

    parameters_before = [
        parameter
        .detach()
        .clone()
        .float()
        for parameter
        in method.trainable_parameters
    ]

    method.optimizer.zero_grad(
        set_to_none=True
    )

    method._assign_update_gradients(
        auxiliary_gradients=(
            aux_gradients
        ),
        anchor_gradients=(
            anchor_gradients
        ),
    )

    if method.gradient_clip > 0.0:

        torch.nn.utils.clip_grad_norm_(
            method.trainable_parameters,
            max_norm=(
                method.gradient_clip
            ),
        )

    method.optimizer.step()

    return float(
        method._parameter_delta_norm(
            parameters=(
                method.trainable_parameters
            ),
            before=(
                parameters_before
            ),
        )
    )


# ============================================================
# FROZEN
# ============================================================


def _evaluate_frozen(
    *,
    model: torch.nn.Module,
    X: np.ndarray,
    y_main: np.ndarray,
    device: torch.device,
    n_samples: int,
) -> dict[str, Any]:

    correct = 0
    losses = []

    model.eval()

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

        y = torch.as_tensor(
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

        prediction = _predict(
            model,
            x,
        )

        correct += int(
            prediction
            == int(
                y.item()
            )
        )

        losses.append(
            _main_loss(
                model,
                x,
                y,
            )
        )

    return {
        "accuracy":
            float(
                correct
                / n_samples
            ),

        "mean_main_loss":
            float(
                np.mean(
                    losses
                )
            ),

        "n_samples":
            int(
                n_samples
            ),
    }


# ============================================================
# V3 TRAJECTORY
# ============================================================


def _run_v3(
    *,
    source_model: torch.nn.Module,
    method_config: dict[str, Any],
    X: np.ndarray,
    y_main: np.ndarray,
    y_aux: np.ndarray,
    device: torch.device,
    experiment_seed: int,
    n_samples: int,
) -> dict[str, Any]:

    config = copy.deepcopy(
        method_config
    )

    # Force exact V3 behavior.
    config[
        "main_preservation_enabled"
    ] = False

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

    correct = 0
    losses = []

    n_updates = 0
    n_gate_pass = 0

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

        result = method.observe(
            x=x,
            y_aux=y_aux_tensor,
        )

        if result.applied:
            n_updates += 1
            n_gate_pass += 1

        prediction = _predict(
            method.model,
            x,
        )

        correct += int(
            prediction
            == int(
                y_main_tensor.item()
            )
        )

        losses.append(
            _main_loss(
                method.model,
                x,
                y_main_tensor,
            )
        )

    return {
        "accuracy":
            float(
                correct
                / n_samples
            ),

        "mean_main_loss":
            float(
                np.mean(
                    losses
                )
            ),

        "n_samples":
            int(
                n_samples
            ),

        "n_gate_pass":
            int(
                n_gate_pass
            ),

        "n_updates":
            int(
                n_updates
            ),

        "update_rate":
            float(
                n_updates
                / n_samples
            ),
    }


# ============================================================
# ORACLE TRAJECTORY
# ============================================================


def _run_oracle(
    *,
    mode: str,
    source_model: torch.nn.Module,
    method_config: dict[str, Any],
    X: np.ndarray,
    y_main: np.ndarray,
    y_aux: np.ndarray,
    device: torch.device,
    experiment_seed: int,
    n_samples: int,
) -> dict[str, Any]:

    if mode not in {
        "gate",
        "projection",
    }:
        raise ValueError(
            mode
        )

    config = copy.deepcopy(
        method_config
    )

    # We are manually controlling gradients.
    config[
        "main_preservation_enabled"
    ] = False

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
            "Oracle diagnostic currently requires steps == 1."
        )

    correct = 0
    losses = []

    n_gate_pass = 0
    n_updates = 0

    n_raw_main_conflicts = 0
    n_oracle_skips = 0
    n_oracle_projections = 0

    sum_aux_main_cosine = 0.0
    sum_parameter_delta = 0.0

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
        # SAME RELIABILITY WINDOW AS CUMULATIVE
        # ----------------------------------------------------

        method.model.eval()

        with torch.no_grad():

            gate_loss = (
                method._auxiliary_loss(
                    x_tensor=x,
                    y_aux_tensor=(
                        y_aux_tensor
                    ),
                )
            )

        normalized_aux_loss = float(
            gate_loss.item()
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

        if gate_pass:

            n_gate_pass += 1

            # -----------------------------------------------
            # RAW AUX GRADIENT
            # -----------------------------------------------

            aux_loss = (
                method._auxiliary_loss(
                    x_tensor=x,
                    y_aux_tensor=(
                        y_aux_tensor
                    ),
                )
            )

            aux_gradients = (
                method._autograd_gradient_list(
                    aux_loss
                )
            )

            # -----------------------------------------------
            # TRUE MAIN GRADIENT — ORACLE DIAGNOSTIC ONLY
            # -----------------------------------------------

            main_logits = (
                method.model(
                    x
                )
            )

            true_main_loss = (
                F.cross_entropy(
                    main_logits,
                    y_main_tensor,
                )
            )

            main_gradients = (
                method._autograd_gradient_list(
                    true_main_loss
                )
            )

            dot_product = (
                _gradient_dot(
                    aux_gradients,
                    main_gradients,
                )
            )

            cosine = (
                _gradient_cosine(
                    aux_gradients,
                    main_gradients,
                    epsilon=(
                        method.epsilon
                    ),
                )
            )

            sum_aux_main_cosine += (
                cosine
            )

            conflict = bool(
                dot_product < 0.0
            )

            if conflict:
                n_raw_main_conflicts += 1

            # -----------------------------------------------
            # ORACLE POLICY
            # -----------------------------------------------

            should_update = True

            accepted_gradients = (
                aux_gradients
            )

            if mode == "gate":

                # Only use metadata updates whose descent
                # direction is first-order compatible with
                # the TRUE main-task loss.
                if dot_product <= 0.0:
                    should_update = False
                    n_oracle_skips += 1

            elif mode == "projection":

                if conflict:

                    accepted_gradients = (
                        _oracle_project(
                            aux_gradients,
                            main_gradients,
                            epsilon=(
                                method.epsilon
                            ),
                        )
                    )

                    n_oracle_projections += 1

            # -----------------------------------------------
            # ACTUAL ADAM UPDATE
            # -----------------------------------------------

            if should_update:

                parameter_delta = (
                    _apply_gradient_update(
                        method=method,
                        aux_gradients=(
                            accepted_gradients
                        ),
                    )
                )

                sum_parameter_delta += (
                    parameter_delta
                )

                n_updates += 1

        # ----------------------------------------------------
        # ADAPT-THEN-PREDICT
        # ----------------------------------------------------

        prediction = _predict(
            method.model,
            x,
        )

        correct += int(
            prediction
            == int(
                y_main_tensor.item()
            )
        )

        losses.append(
            _main_loss(
                method.model,
                x,
                y_main_tensor,
            )
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
                f"[oracle-{mode}] "
                f"{index + 1}/{n_samples}"
            )

    denominator = max(
        n_gate_pass,
        1,
    )

    update_denominator = max(
        n_updates,
        1,
    )

    return {
        "accuracy":
            float(
                correct
                / n_samples
            ),

        "mean_main_loss":
            float(
                np.mean(
                    losses
                )
            ),

        "n_samples":
            int(
                n_samples
            ),

        "n_gate_pass":
            int(
                n_gate_pass
            ),

        "gate_pass_rate":
            float(
                n_gate_pass
                / n_samples
            ),

        "n_updates":
            int(
                n_updates
            ),

        "update_rate":
            float(
                n_updates
                / n_samples
            ),

        "raw_main_conflict_count":
            int(
                n_raw_main_conflicts
            ),

        "raw_main_conflict_rate_gate_pass":
            float(
                n_raw_main_conflicts
                / denominator
            ),

        "oracle_skip_count":
            int(
                n_oracle_skips
            ),

        "oracle_projection_count":
            int(
                n_oracle_projections
            ),

        "mean_aux_main_cosine_gate_pass":
            float(
                sum_aux_main_cosine
                / denominator
            ),

        "mean_parameter_delta_per_update":
            float(
                sum_parameter_delta
                / update_denominator
            ),
    }


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

    config = _load_config(
        run_directory
    )

    if (
        config.dataset_name
        != "huffpost"
    ):
        raise RuntimeError(
            "This diagnostic is restricted to HuffPost."
        )

    device = _device(
        args.device
    )

    print()
    print("=" * 80)
    print(
        "HUFFPOST ORACLE METADATA-ADAPTATION DIAGNOSTIC"
    )
    print("=" * 80)
    print(
        f"run_directory={run_directory}"
    )
    print(
        f"experiment_seed={config.experiment_seed}"
    )
    print(
        f"split_seed={config.split_seed}"
    )
    print(
        f"device={device}"
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
            "Expected pseudo-source 2012-2014."
        )

    if tuple(
        protocol.years.pseudo_ood
    ) != (
        2015,
    ):
        raise RuntimeError(
            "Expected pseudo-OOD year 2015."
        )

    # ========================================================
    # RECREATE THE TUNING SOURCE MODEL ONCE
    # ========================================================

    print()
    print("=" * 80)
    print(
        "RECONSTRUCTING TUNING SOURCE MODEL"
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
    # 2015 VALIDATION STREAM
    # ========================================================

    pseudo_ood = (
        protocol
        .pseudo_ood_validation_stream()
    )

    if len(
        pseudo_ood
    ) != 1:
        raise RuntimeError(
            "Expected exactly one pseudo-OOD validation slice."
        )

    stream = pseudo_ood[
        0
    ]

    if int(
        stream.year
    ) != 2015:
        raise RuntimeError(
            f"Expected 2015, got {stream.year}."
        )

    n_samples = len(
        stream.X
    )

    if args.max_samples > 0:

        n_samples = min(
            n_samples,
            int(
                args.max_samples
            ),
        )

    cumulative_config = dict(
        config.method_config(
            "metadata_cumulative"
        )
    )

    print()
    print(
        f"2015 validation samples={n_samples}"
    )
    print(
        "cumulative learning_rate="
        f"{cumulative_config['learning_rate']}"
    )
    print(
        "window=["
        f"{cumulative_config['normalized_aux_loss_min']}, "
        f"{cumulative_config['normalized_aux_loss_max']}]"
    )
    print(
        "regularization="
        f"{cumulative_config['regularization']}"
    )

    # ========================================================
    # FOUR TRAJECTORIES
    # ========================================================

    print()
    print("=" * 80)
    print("FROZEN")
    print("=" * 80)

    frozen = _evaluate_frozen(
        model=source_double,
        X=stream.X,
        y_main=stream.y_main,
        device=device,
        n_samples=n_samples,
    )

    print(
        json.dumps(
            frozen,
            indent=2,
        )
    )

    print()
    print("=" * 80)
    print("V3 RAW CUMULATIVE")
    print("=" * 80)

    v3 = _run_v3(
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
        n_samples=n_samples,
    )

    print(
        json.dumps(
            v3,
            indent=2,
        )
    )

    print()
    print("=" * 80)
    print("ORACLE GATE")
    print("=" * 80)

    oracle_gate = _run_oracle(
        mode="gate",
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
        n_samples=n_samples,
    )

    print(
        json.dumps(
            oracle_gate,
            indent=2,
        )
    )

    print()
    print("=" * 80)
    print("ORACLE PROJECTION")
    print("=" * 80)

    oracle_projection = _run_oracle(
        mode="projection",
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
        n_samples=n_samples,
    )

    print(
        json.dumps(
            oracle_projection,
            indent=2,
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "dataset":
            "huffpost",

        "year":
            2015,

        "split":
            "validation",

        "experiment_seed":
            int(
                config.experiment_seed
            ),

        "split_seed":
            int(
                config.split_seed
            ),

        "uses_main_labels_in_real_tta":
            False,

        "oracle_note":
            (
                "2015 y_main is used only by the oracle "
                "diagnostic to establish an upper bound on "
                "metadata-gradient selection/projection."
            ),

        "frozen":
            frozen,

        "v3_raw":
            v3,

        "oracle_gate":
            oracle_gate,

        "oracle_projection":
            oracle_projection,

        "delta_accuracy_vs_frozen":
            {
                "v3_raw":
                    float(
                        v3[
                            "accuracy"
                        ]
                        - frozen[
                            "accuracy"
                        ]
                    ),

                "oracle_gate":
                    float(
                        oracle_gate[
                            "accuracy"
                        ]
                        - frozen[
                            "accuracy"
                        ]
                    ),

                "oracle_projection":
                    float(
                        oracle_projection[
                            "accuracy"
                        ]
                        - frozen[
                            "accuracy"
                        ]
                    ),
            },

        "delta_accuracy_vs_v3":
            {
                "oracle_gate":
                    float(
                        oracle_gate[
                            "accuracy"
                        ]
                        - v3[
                            "accuracy"
                        ]
                    ),

                "oracle_projection":
                    float(
                        oracle_projection[
                            "accuracy"
                        ]
                        - v3[
                            "accuracy"
                        ]
                    ),
            },
    }

    output_directory = (
        run_directory
        / "analysis"
        / "huffpost_oracle_alignment"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / "summary.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
            sort_keys=True,
        )

    print()
    print("=" * 80)
    print("FINAL ORACLE COMPARISON")
    print("=" * 80)

    print(
        f"Frozen:            "
        f"{100.0 * frozen['accuracy']:.3f}%"
    )

    print(
        f"V3 raw:            "
        f"{100.0 * v3['accuracy']:.3f}% "
        f"({100.0 * summary['delta_accuracy_vs_frozen']['v3_raw']:+.3f} pp)"
    )

    print(
        f"Oracle gate:        "
        f"{100.0 * oracle_gate['accuracy']:.3f}% "
        f"({100.0 * summary['delta_accuracy_vs_frozen']['oracle_gate']:+.3f} pp)"
    )

    print(
        f"Oracle projection:  "
        f"{100.0 * oracle_projection['accuracy']:.3f}% "
        f"({100.0 * summary['delta_accuracy_vs_frozen']['oracle_projection']:+.3f} pp)"
    )

    print()
    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()