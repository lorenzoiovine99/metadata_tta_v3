from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_ROOT),
    )


from metadata_tta.config import ExperimentConfig
from metadata_tta.data import build_data_bundle
from metadata_tta.protocols import V3Protocol
from metadata_tta.reproducibility import set_seed
from metadata_tta.tta import (
    EvaluationOrder,
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
            "Paired Frozen-vs-Episodic flip diagnostic "
            "on HuffPost pseudo-OOD validation 2015."
        )
    )

    parser.add_argument(
        "--run-directory",
        required=True,
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

    return parser.parse_args()


# ============================================================
# DEVICE / CONFIG
# ============================================================


def _device(
    requested: str,
) -> torch.device:

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        return torch.device("cuda")

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

        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(
            torch.backends,
            "mps",
        )
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


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
# PREDICTION METRICS
# ============================================================


def _main_reference_metrics(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    y_main: int,
) -> dict[str, float | int]:

    model.eval()

    with torch.no_grad():

        logits = model(
            x
        )

        probabilities = F.softmax(
            logits,
            dim=1,
        )

        log_probabilities = torch.log(
            probabilities.clamp_min(
                1.0e-12
            )
        )

        entropy = -torch.sum(
            probabilities
            * log_probabilities,
            dim=1,
        )

        n_classes = int(
            logits.shape[1]
        )

        normalized_confidence = (
            1.0
            - entropy
            / math.log(
                n_classes
            )
        )

        top_values, top_indices = (
            logits.topk(
                k=2,
                dim=1,
            )
        )

        prediction = int(
            top_indices[
                0,
                0,
            ].item()
        )

        margin = float(
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
        )

        true_probability = float(
            probabilities[
                0,
                int(y_main),
            ].item()
        )

        loss = float(
            F.cross_entropy(
                logits,
                torch.as_tensor(
                    [
                        int(
                            y_main
                        )
                    ],
                    dtype=torch.long,
                    device=x.device,
                ),
            ).item()
        )

    return {
        "prediction":
            prediction,

        "confidence":
            float(
                normalized_confidence[
                    0
                ].item()
            ),

        "margin":
            margin,

        "true_probability":
            true_probability,

        "loss":
            loss,
    }


def _aux_reference_metrics(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    y_aux: int,
) -> dict[str, float | int]:

    model.eval()

    with torch.no_grad():

        logits = model.forward_aux(
            x
        )

        probabilities = F.softmax(
            logits,
            dim=1,
        )

        top_values, top_indices = (
            logits.topk(
                k=2,
                dim=1,
            )
        )

        prediction = int(
            top_indices[
                0,
                0,
            ].item()
        )

        confidence = float(
            probabilities.max(
                dim=1
            ).values[
                0
            ].item()
        )

        margin = float(
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
        )

        true_probability = float(
            probabilities[
                0,
                int(y_aux),
            ].item()
        )

        loss = float(
            F.cross_entropy(
                logits,
                torch.as_tensor(
                    [
                        int(
                            y_aux
                        )
                    ],
                    dtype=torch.long,
                    device=x.device,
                ),
            ).item()
        )

    return {
        "prediction":
            prediction,

        "confidence":
            confidence,

        "margin":
            margin,

        "true_probability":
            true_probability,

        "loss":
            loss,
    }


def _adapted_main_metrics(
    *,
    method,
    x: np.ndarray,
    y_main: int,
) -> dict[str, float | int]:

    logits = method.predict_logits(
        x
    )

    probabilities = F.softmax(
        logits,
        dim=1,
    )

    prediction = int(
        logits.argmax(
            dim=1
        )[0].item()
    )

    y_tensor = torch.as_tensor(
        [
            int(
                y_main
            )
        ],
        dtype=torch.long,
        device=logits.device,
    )

    loss = float(
        F.cross_entropy(
            logits,
            y_tensor,
        )
        .detach()
        .item()
    )

    true_probability = float(
        probabilities[
            0,
            int(y_main),
        ]
        .detach()
        .item()
    )

    return {
        "prediction":
            prediction,

        "true_probability":
            true_probability,

        "loss":
            loss,
    }


# ============================================================
# TRANSITIONS
# ============================================================


def _transition(
    *,
    frozen_correct: bool,
    tta_correct: bool,
) -> str:

    if frozen_correct and tta_correct:
        return "correct_to_correct"

    if frozen_correct and not tta_correct:
        return "correct_to_wrong"

    if not frozen_correct and tta_correct:
        return "wrong_to_correct"

    return "wrong_to_wrong"


# ============================================================
# AUC
# ============================================================


def _binary_auc(
    values: pd.Series,
    labels: pd.Series,
) -> float | None:
    """
    Mann-Whitney / rank-based ROC AUC.

    label=1 means beneficial flip:
        Frozen wrong -> TTA correct

    label=0 means harmful flip:
        Frozen correct -> TTA wrong
    """

    frame = pd.DataFrame(
        {
            "value":
                pd.to_numeric(
                    values,
                    errors="coerce",
                ),
            "label":
                labels.astype(
                    int
                ),
        }
    ).dropna()

    if len(frame) == 0:
        return None

    n_positive = int(
        (
            frame[
                "label"
            ]
            == 1
        ).sum()
    )

    n_negative = int(
        (
            frame[
                "label"
            ]
            == 0
        ).sum()
    )

    if (
        n_positive == 0
        or n_negative == 0
    ):
        return None

    ranks = (
        frame[
            "value"
        ]
        .rank(
            method="average"
        )
    )

    rank_sum_positive = float(
        ranks[
            frame[
                "label"
            ]
            == 1
        ].sum()
    )

    auc = (
        rank_sum_positive
        - n_positive
        * (
            n_positive
            + 1
        )
        / 2.0
    ) / (
        n_positive
        * n_negative
    )

    return float(
        auc
    )


# ============================================================
# SUMMARIES
# ============================================================


OBSERVABLE_FEATURES = [
    "normalized_aux_loss",
    "aux_loss",
    "source_confidence",
    "source_margin",
    "aux_confidence",
    "aux_margin",
    "aux_true_probability",
    "gradient_norm",
    "parameter_delta",
]


def _safe_mean(
    series: pd.Series,
) -> float | None:

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return None

    return float(
        values.mean()
    )


def _safe_median(
    series: pd.Series,
) -> float | None:

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return None

    return float(
        values.median()
    )


def _feature_summary(
    frame: pd.DataFrame,
) -> dict[str, Any]:

    result: dict[
        str,
        Any,
    ] = {
        "n":
            int(
                len(frame)
            )
    }

    for feature in (
        OBSERVABLE_FEATURES
    ):

        result[
            feature
        ] = {
            "mean":
                _safe_mean(
                    frame[
                        feature
                    ]
                ),

            "median":
                _safe_median(
                    frame[
                        feature
                    ]
                ),
        }

    return result


# ============================================================
# MAIN DIAGNOSTIC
# ============================================================


def main() -> None:

    args = _parse_args()

    run_directory = (
        Path(
            args.run_directory
        )
        .expanduser()
        .resolve()
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
        "HUFFPOST EPISODIC PAIRED-FLIP DIAGNOSTIC"
    )
    print("=" * 80)
    print(
        f"run_directory={run_directory}"
    )
    print(
        f"experiment_seed="
        f"{config.experiment_seed}"
    )
    print(
        f"split_seed="
        f"{config.split_seed}"
    )
    print(
        f"device={device}"
    )

    # ========================================================
    # DATA / PROTOCOL
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
            "Expected pseudo-OOD 2015."
        )

    # ========================================================
    # RECONSTRUCT PSEUDO-SOURCE
    # ========================================================

    print()
    print("=" * 80)
    print(
        "RECONSTRUCTING PSEUDO-SOURCE 2012-2014"
    )
    print("=" * 80)

    (
        source_single,
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

    source_single.eval()
    source_double.eval()

    # ========================================================
    # PSEUDO-OOD VALIDATION 2015
    # ========================================================

    slices = (
        protocol
        .pseudo_ood_validation_stream()
    )

    if len(
        slices
    ) != 1:
        raise RuntimeError(
            "Expected exactly one pseudo-OOD slice."
        )

    stream = slices[
        0
    ]

    if int(
        stream.year
    ) != 2015:
        raise RuntimeError(
            f"Expected 2015, got {stream.year}."
        )

    # ========================================================
    # EPISODIC METHOD
    #
    # Force surgery OFF. We want to study only the metadata
    # update produced by this Aux head.
    # ========================================================

    method_config = dict(
        config.method_config(
            "metadata_episodic"
        )
    )

    method_config[
        "main_preservation_enabled"
    ] = False

    print()
    print(
        "EPISODIC CONFIG"
    )

    print(
        json.dumps(
            method_config,
            indent=2,
            default=str,
        )
    )

    set_seed(
        config.experiment_seed
    )

    method_class = get_method_class(
        "metadata_episodic"
    )

    method = method_class(
        source_model=copy.deepcopy(
            source_double
        ),
        config=method_config,
        device=device,
    )

    if (
        method.evaluation_order
        is not
        EvaluationOrder.ADAPT_THEN_PREDICT
    ):
        raise RuntimeError(
            "Expected metadata_episodic to use "
            "ADAPT_THEN_PREDICT."
        )

    method.on_year_start(
        2015
    )

    # ========================================================
    # SAMPLE-WISE PAIRED ANALYSIS
    # ========================================================

    rows: list[
        dict[str, Any]
    ] = []

    n_samples = int(
        len(
            stream.X
        )
    )

    for index in range(
        n_samples
    ):

        x_numpy = np.asarray(
            stream.X[
                index
            ],
            dtype=np.float32,
        )

        y_main = int(
            stream.y_main[
                index
            ]
        )

        y_aux = int(
            stream.y_aux[
                index
            ]
        )

        x_tensor = torch.as_tensor(
            x_numpy[
                None,
                :
            ],
            dtype=torch.float32,
            device=device,
        )

        # ----------------------------------------------------
        # IMMUTABLE FROZEN SOURCE
        # ----------------------------------------------------

        source_main = (
            _main_reference_metrics(
                model=source_double,
                x=x_tensor,
                y_main=y_main,
            )
        )

        source_aux = (
            _aux_reference_metrics(
                model=source_double,
                x=x_tensor,
                y_aux=y_aux,
            )
        )

        frozen_prediction = int(
            source_main[
                "prediction"
            ]
        )

        frozen_correct = bool(
            frozen_prediction
            == y_main
        )

        # ----------------------------------------------------
        # EXACT EVALUATOR ORDER:
        # OBSERVE -> PREDICT
        # ----------------------------------------------------

        adaptation_result = (
            method.observe(
                x=x_numpy,
                y_aux=y_aux,
            )
        )

        adapted_main = (
            _adapted_main_metrics(
                method=method,
                x=x_numpy,
                y_main=y_main,
            )
        )

        tta_prediction = int(
            adapted_main[
                "prediction"
            ]
        )

        tta_correct = bool(
            tta_prediction
            == y_main
        )

        transition = _transition(
            frozen_correct=frozen_correct,
            tta_correct=tta_correct,
        )

        diagnostics = dict(
            adaptation_result.diagnostics
        )

        row = {
            "year":
                2015,

            "sample_index":
                int(
                    index
                ),

            "y_main":
                y_main,

            "y_aux":
                y_aux,

            "frozen_prediction":
                frozen_prediction,

            "tta_prediction":
                tta_prediction,

            "frozen_correct":
                frozen_correct,

            "tta_correct":
                tta_correct,

            "prediction_changed":
                bool(
                    frozen_prediction
                    != tta_prediction
                ),

            "transition":
                transition,

            "beneficial_flip":
                bool(
                    transition
                    == "wrong_to_correct"
                ),

            "harmful_flip":
                bool(
                    transition
                    == "correct_to_wrong"
                ),

            "update_applied":
                bool(
                    adaptation_result.applied
                ),

            "update_reason":
                str(
                    diagnostics.get(
                        "reason",
                        ""
                    )
                ),

            # -----------------------------------------------
            # TEST-TIME OBSERVABLES
            # -----------------------------------------------

            "aux_loss":
                float(
                    diagnostics.get(
                        "aux_loss",
                        source_aux[
                            "loss"
                        ],
                    )
                ),

            "normalized_aux_loss":
                float(
                    diagnostics.get(
                        "normalized_aux_loss",
                        float(
                            "nan"
                        ),
                    )
                ),

            "source_confidence":
                float(
                    source_main[
                        "confidence"
                    ]
                ),

            "source_margin":
                float(
                    source_main[
                        "margin"
                    ]
                ),

            "aux_prediction":
                int(
                    source_aux[
                        "prediction"
                    ]
                ),

            "aux_prediction_correct":
                bool(
                    int(
                        source_aux[
                            "prediction"
                        ]
                    )
                    == y_aux
                ),

            "aux_confidence":
                float(
                    source_aux[
                        "confidence"
                    ]
                ),

            "aux_margin":
                float(
                    source_aux[
                        "margin"
                    ]
                ),

            "aux_true_probability":
                float(
                    source_aux[
                        "true_probability"
                    ]
                ),

            "gradient_norm":
                float(
                    diagnostics.get(
                        "gradient_norm",
                        float(
                            "nan"
                        ),
                    )
                ),

            "parameter_delta":
                float(
                    diagnostics.get(
                        "parameter_delta",
                        float(
                            "nan"
                        ),
                    )
                ),

            # -----------------------------------------------
            # Y_MAIN DIAGNOSTIC ONLY
            # NEVER VALID TTA INPUTS
            # -----------------------------------------------

            "frozen_main_loss":
                float(
                    source_main[
                        "loss"
                    ]
                ),

            "tta_main_loss":
                float(
                    adapted_main[
                        "loss"
                    ]
                ),

            "main_loss_delta":
                float(
                    adapted_main[
                        "loss"
                    ]
                    - source_main[
                        "loss"
                    ]
                ),

            "frozen_true_probability":
                float(
                    source_main[
                        "true_probability"
                    ]
                ),

            "tta_true_probability":
                float(
                    adapted_main[
                        "true_probability"
                    ]
                ),

            "true_probability_delta":
                float(
                    adapted_main[
                        "true_probability"
                    ]
                    - source_main[
                        "true_probability"
                    ]
                ),
        }

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
                f"[FLIPS] "
                f"{index + 1}/{n_samples}"
            )

    frame = pd.DataFrame(
        rows
    )

    # ========================================================
    # COUNTS
    # ========================================================

    transition_order = [
        "correct_to_correct",
        "correct_to_wrong",
        "wrong_to_correct",
        "wrong_to_wrong",
    ]

    transition_counts = {
        transition:
            int(
                (
                    frame[
                        "transition"
                    ]
                    == transition
                )
                .sum()
            )
        for transition
        in transition_order
    }

    harmful = int(
        transition_counts[
            "correct_to_wrong"
        ]
    )

    beneficial = int(
        transition_counts[
            "wrong_to_correct"
        ]
    )

    total_flips = int(
        frame[
            "prediction_changed"
        ].sum()
    )

    # ========================================================
    # FEATURE STATISTICS BY TRANSITION
    # ========================================================

    transition_feature_summary = {
        transition:
            _feature_summary(
                frame[
                    frame[
                        "transition"
                    ]
                    == transition
                ]
            )
        for transition
        in transition_order
    }

    # ========================================================
    # HARMFUL VS BENEFICIAL SEPARABILITY
    # ========================================================

    flips = frame[
        frame[
            "transition"
        ].isin(
            [
                "correct_to_wrong",
                "wrong_to_correct",
            ]
        )
    ].copy()

    flips[
        "beneficial_label"
    ] = (
        flips[
            "transition"
        ]
        == "wrong_to_correct"
    ).astype(
        int
    )

    aucs: dict[
        str,
        Any,
    ] = {}

    for feature in (
        OBSERVABLE_FEATURES
    ):

        auc = _binary_auc(
            flips[
                feature
            ],
            flips[
                "beneficial_label"
            ],
        )

        aucs[
            feature
        ] = {
            "auc_beneficial":
                auc,

            "direction_free_separation":
                (
                    None
                    if auc is None
                    else float(
                        max(
                            auc,
                            1.0 - auc,
                        )
                    )
                ),
        }

    # ========================================================
    # AUX LABEL DISTRIBUTIONS
    # ========================================================

    aux_label_distribution = {}

    for transition in (
        [
            "correct_to_wrong",
            "wrong_to_correct",
        ]
    ):

        subset = frame[
            frame[
                "transition"
            ]
            == transition
        ]

        counts = (
            subset[
                "y_aux"
            ]
            .value_counts(
                normalize=False
            )
            .sort_index()
        )

        aux_label_distribution[
            transition
        ] = {
            str(
                int(
                    label
                )
            ):
                int(
                    count
                )
            for label, count
            in counts.items()
        }

    # ========================================================
    # MAIN LABEL DISTRIBUTIONS
    # ========================================================

    main_label_distribution = {}

    for transition in (
        [
            "correct_to_wrong",
            "wrong_to_correct",
        ]
    ):

        subset = frame[
            frame[
                "transition"
            ]
            == transition
        ]

        counts = (
            subset[
                "y_main"
            ]
            .value_counts(
                normalize=False
            )
            .sort_index()
        )

        main_label_distribution[
            transition
        ] = {
            str(
                int(
                    label
                )
            ):
                int(
                    count
                )
            for label, count
            in counts.items()
        }

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "dataset":
            "huffpost",

        "year":
            2015,

        "split":
            "pseudo_ood_validation",

        "experiment_seed":
            int(
                config.experiment_seed
            ),

        "split_seed":
            int(
                config.split_seed
            ),

        "main_preservation_forced_off":
            True,

        "n_samples":
            n_samples,

        "frozen_accuracy":
            float(
                frame[
                    "frozen_correct"
                ].mean()
            ),

        "episodic_accuracy":
            float(
                frame[
                    "tta_correct"
                ].mean()
            ),

        "delta_accuracy":
            float(
                frame[
                    "tta_correct"
                ].mean()
                - frame[
                    "frozen_correct"
                ].mean()
            ),

        "n_updates":
            int(
                frame[
                    "update_applied"
                ].sum()
            ),

        "update_rate":
            float(
                frame[
                    "update_applied"
                ].mean()
            ),

        "prediction_changed_count":
            total_flips,

        "prediction_changed_rate":
            float(
                frame[
                    "prediction_changed"
                ].mean()
            ),

        "transition_counts":
            transition_counts,

        "beneficial_flip_count":
            beneficial,

        "harmful_flip_count":
            harmful,

        "net_correct_flip_balance":
            int(
                beneficial
                - harmful
            ),

        "transition_feature_summary":
            transition_feature_summary,

        "beneficial_vs_harmful_observable_auc":
            aucs,

        "aux_label_distribution_on_flips":
            aux_label_distribution,

        "main_label_distribution_on_flips":
            main_label_distribution,

        "mean_main_loss_delta":
            float(
                frame[
                    "main_loss_delta"
                ].mean()
            ),

        "mean_true_probability_delta":
            float(
                frame[
                    "true_probability_delta"
                ].mean()
            ),
    }

    # ========================================================
    # SAVE
    # ========================================================

    output_directory = (
        run_directory
        / "analysis"
        / "huffpost_episodic_flips"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_directory
        / "samples.csv"
    )

    summary_path = (
        output_directory
        / "summary.json"
    )

    frame.to_csv(
        csv_path,
        index=False,
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
            sort_keys=True,
        )

    # ========================================================
    # PRINT COMPACT RESULT
    # ========================================================

    print()
    print("=" * 80)
    print(
        "PAIRED FLIP RESULT"
    )
    print("=" * 80)

    print(
        f"Frozen accuracy:   "
        f"{100.0 * summary['frozen_accuracy']:.3f}%"
    )

    print(
        f"Episodic accuracy: "
        f"{100.0 * summary['episodic_accuracy']:.3f}%"
    )

    print(
        f"Delta:             "
        f"{100.0 * summary['delta_accuracy']:+.3f} pp"
    )

    print()

    print(
        "correct -> wrong: "
        f"{harmful}"
    )

    print(
        "wrong -> correct: "
        f"{beneficial}"
    )

    print(
        "net flip balance: "
        f"{beneficial - harmful:+d}"
    )

    print(
        "prediction changes: "
        f"{total_flips}/{n_samples}"
    )

    print()
    print(
        "OBSERVABLE SEPARATION AUC "
        "(beneficial vs harmful flips)"
    )

    for feature in (
        OBSERVABLE_FEATURES
    ):

        result = aucs[
            feature
        ]

        print(
            f"{feature:28s} "
            f"auc={result['auc_beneficial']} "
            f"sep={result['direction_free_separation']}"
        )

    print()
    print(
        f"Saved samples: {csv_path}"
    )

    print(
        f"Saved summary: {summary_path}"
    )


if __name__ == "__main__":
    main()