from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

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
    evaluate_tta_stream,
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
    initialize_double_from_single,
    train_aux_head_only,
)
from metadata_tta.tta import (
    get_method_class,
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
    load_best_result,
    save_best_result,
    save_trial_rows,
)
from .search import (
    build_search_strategy,
)


@dataclass(frozen=True)
class TTATuningResult:
    method_name: str
    best_score: float
    best_parameters: dict[str, Any]
    best_overrides: dict[str, Any]
    output_directory: Path


# ============================================================
# MODEL FAMILY
# ============================================================


def _method_family(
    method_name: str,
) -> str:

    if method_name == "tent":
        return "single_head"

    if method_name in {
        "metadata",
        "temporal_gradient",
        "consensus",
        "experimental",
    }:
        return "double_head"

    raise ValueError(
        f"Unsupported tuning method: "
        f"{method_name}"
    )


# ============================================================
# YEAR SPLIT
# ============================================================


def _split_for_year(
    bundle: DataBundle,
    config: ExperimentConfig,
    year: int,
):

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

    return (
        data,
        split,
    )


# ============================================================
# SOURCE TRAINING DATA
# ============================================================

def _source_supervised_arrays(
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

    data, split = (
        _split_for_year(
            bundle=bundle,
            config=config,
            year=year,
        )
    )

    if len(
        split.validation
    ) == 0:
        raise ValueError(
            f"TTA source year {year} has an empty "
            "validation split."
        )

    return (
        # TRAIN
        data.X[
            split.train
        ],

        data.y_main[
            split.train
        ],

        data.y_aux[
            split.train
        ],

        # VALIDATION
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
# SOURCE MODEL
# ============================================================

def _train_source_model(
    *,
    family: str,
    config: ExperimentConfig,
    bundle: DataBundle,
    source_years: list[int],
    device: torch.device,
) -> nn.Module:

    if not source_years:
        raise ValueError(
            "TTA tuning requires at least "
            "one supervised source year."
        )

    if family == "single_head":

        model = (
            create_single_head_model(
                input_dim=bundle.feature_dim,
                config=config,
            )
            .to(device)
        )

    elif family == "double_head":

        model = (
            create_double_head_model(
                input_dim=bundle.feature_dim,
                config=config,
            )
            .to(device)
        )

    else:

        raise ValueError(
            f"Unknown family: {family}"
        )

    # Reset RNG after architecture-specific model construction.
    # This makes DataLoader shuffling / dropout randomness
    # independent of the extra auxiliary-head initialization.
    set_seed(config.seed)

    optimizer = None

    aux_loss_weight = (
        get_aux_loss_weight(
            config
        )
        if family == "double_head"
        else None
    )

    for index, year in enumerate(
        source_years
    ):

        (
            X_train,
            y_main_train,
            y_aux_train,
            X_validation,
            y_main_validation,
            y_aux_validation,
        ) = _source_supervised_arrays(
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
                model_kind=family,
            )
        )

        if family == "single_head":

            result = train_single_head(
                model=model,

                X=X_train,

                y_main=(
                    y_main_train
                ),

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
                    f"TTA tuning source {year} | "
                    "Single Head"
                ),
            )

        else:

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

                aux_loss_weight=float(
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
                    f"TTA tuning source {year} | "
                    "Double Head"
                ),
            )

        model = (
            result.model
        )

        optimizer = (
            result.optimizer
        )

    model.eval()

    return model

def _train_aux_only_double_from_single(
    *,
    source_single: nn.Module,
    config: ExperimentConfig,
    bundle: DataBundle,
    source_years: list[int],
    device: torch.device,
) -> nn.Module:
    """
    Build a double-head model whose complete main-task path
    is copied from the trained single-head source model.

    Then train ONLY the auxiliary head using source metadata.
    """

    if not source_years:
        raise ValueError(
            "Aux-only double training requires "
            "at least one source year."
        )

    # --------------------------------------------------------
    # 1. Create double architecture.
    # --------------------------------------------------------

    set_seed(config.seed)

    model = (
        create_double_head_model(
            input_dim=bundle.feature_dim,
            config=config,
        )
        .to(device)
    )

    # --------------------------------------------------------
    # 2. Copy the complete trained main path from single.
    # --------------------------------------------------------

    model = initialize_double_from_single(
        single_model=source_single,
        double_model=model,
    )

    # --------------------------------------------------------
    # 3. Gather source-only data for auxiliary-head training.
    # --------------------------------------------------------

    X_train_parts = []
    y_aux_train_parts = []

    X_validation_parts = []
    y_aux_validation_parts = []

    for year in source_years:

        (
            X_train,
            _,
            y_aux_train,
            X_validation,
            _,
            y_aux_validation,
        ) = _source_supervised_arrays(
            bundle=bundle,
            config=config,
            year=year,
        )

        X_train_parts.append(
            X_train
        )

        y_aux_train_parts.append(
            y_aux_train
        )

        X_validation_parts.append(
            X_validation
        )

        y_aux_validation_parts.append(
            y_aux_validation
        )

    X_train_all = np.concatenate(
        X_train_parts,
        axis=0,
    )

    y_aux_train_all = np.concatenate(
        y_aux_train_parts,
        axis=0,
    )

    X_validation_all = np.concatenate(
        X_validation_parts,
        axis=0,
    )

    y_aux_validation_all = np.concatenate(
        y_aux_validation_parts,
        axis=0,
    )

    # --------------------------------------------------------
    # 4. Read the same basic training hyperparameters already
    #    selected for the double-head configuration.
    # --------------------------------------------------------

    training_config = config.section(
        "training"
    )

    base_training = training_config[
        "base"
    ]

    double_training = training_config.get(
        "double_head",
        {},
    )

    base_early_stopping = base_training.get(
        "early_stopping",
        {},
    )

    double_early_stopping = double_training.get(
        "early_stopping",
        {},
    )

    learning_rate = float(
        double_training.get(
            "learning_rate",
            base_training["learning_rate"],
        )
    )

    weight_decay = float(
        double_training.get(
            "weight_decay",
            base_training["weight_decay"],
        )
    )

    batch_size = int(
        double_training.get(
            "batch_size",
            base_training["batch_size"],
        )
    )

    epochs = int(
        double_training.get(
            "epochs",
            base_training["epochs"],
        )
    )

    patience = int(
        double_early_stopping.get(
            "patience",
            base_early_stopping.get(
                "patience",
                5,
            ),
        )
    )

    min_delta = float(
        double_early_stopping.get(
            "min_delta",
            base_early_stopping.get(
                "min_delta",
                0.0,
            ),
        )
    )

    # Make aux-head training randomness reproducible and
    # independent of double-head initialization.
    set_seed(config.seed)

    model = train_aux_head_only(
        model=model,
        X=X_train_all,
        y_aux=y_aux_train_all,
        X_validation=X_validation_all,
        y_aux_validation=y_aux_validation_all,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        min_delta=min_delta,
        device=device,
        log_prefix="TTA source | AUX-ONLY",
    )

    model.eval()

    return model

# ============================================================
# CONTINUOUS PSEUDO-OOD STREAM
# ============================================================


def _build_pseudo_stream(
    *,
    bundle: DataBundle,
    config: ExperimentConfig,
    years: list[int],
) -> list[
    tuple[
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
]:

    stream = []

    for year in years:

        data, split = (
            _split_for_year(
                bundle=bundle,
                config=config,
                year=year,
            )
        )

        # The split itself is stratified, but once selected the
        # samples are restored to their original relative
        # temporal order.
        indices = np.sort(
            split.test
        )

        stream.append(
            (
                int(year),
                data.X[
                    indices
                ],
                data.y_main[
                    indices
                ],
                data.y_aux[
                    indices
                ],
            )
        )

    return stream

def _print_gradient_alignment_diagnostics(
    *,
    model: nn.Module,
    stream: list[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ],
    device: torch.device,
) -> None:
    """
    Diagnostic only.

    Compare, sample by sample, the main-task gradient and the
    metadata gradient with respect to the online adapter.

    Main labels are used ONLY for analysis here. No update is
    performed and no main label is exposed to TTA.
    """

    if not hasattr(model, "online_adapter"):
        raise TypeError(
            "Gradient diagnostic requires online_adapter."
        )

    if not hasattr(model, "forward_both"):
        raise TypeError(
            "Gradient diagnostic requires forward_both()."
        )

    was_training = model.training
    model.eval()

    adapter_parameters = [
        parameter
        for parameter
        in model.online_adapter.parameters()
        if parameter.requires_grad
    ]

    if not adapter_parameters:
        raise RuntimeError(
            "Gradient diagnostic found no trainable "
            "online_adapter parameters."
        )

    epsilon = 1.0e-12

    for (
        year,
        X,
        y_main,
        y_aux,
    ) in stream:

        cosines: list[float] = []
        main_gradient_norms: list[float] = []
        aux_gradient_norms: list[float] = []

        zero_norm_count = 0

        for index in range(len(X) - 1):

            # Metadata gradient comes from the CURRENT sample t.
            x_tensor = torch.as_tensor(
                X[index:index + 1],
                dtype=torch.float32,
                device=device,
            )

            y_aux_tensor = torch.as_tensor(
                y_aux[index:index + 1],
                dtype=torch.long,
                device=device,
            )

            # Main-task gradient comes from the NEXT sample t+1.
            # Its label is used ONLY for offline diagnosis.
            x_next_tensor = torch.as_tensor(
                X[index + 1:index + 2],
                dtype=torch.float32,
                device=device,
            )

            y_main_next_tensor = torch.as_tensor(
                y_main[index + 1:index + 2],
                dtype=torch.long,
                device=device,
            )

            _, aux_logits = model.forward_both(
                x_tensor
            )

            main_logits_next, _ = model.forward_both(
                x_next_tensor
            )

            aux_loss = (
                torch.nn.functional.cross_entropy(
                    aux_logits,
                    y_aux_tensor,
                )
            )

            main_loss = (
                torch.nn.functional.cross_entropy(
                    main_logits_next,
                    y_main_next_tensor,
                )
            )

            main_gradients = torch.autograd.grad(
                main_loss,
                adapter_parameters,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )

            aux_gradients = torch.autograd.grad(
                aux_loss,
                adapter_parameters,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )

            dot_product = 0.0
            main_norm_squared = 0.0
            aux_norm_squared = 0.0

            for (
                main_gradient,
                aux_gradient,
            ) in zip(
                main_gradients,
                aux_gradients,
            ):

                main_gradient = (
                    main_gradient.detach().float()
                )

                aux_gradient = (
                    aux_gradient.detach().float()
                )

                dot_product += float(
                    torch.sum(
                        main_gradient
                        * aux_gradient
                    ).item()
                )

                main_norm_squared += float(
                    torch.sum(
                        main_gradient ** 2
                    ).item()
                )

                aux_norm_squared += float(
                    torch.sum(
                        aux_gradient ** 2
                    ).item()
                )

            main_norm = float(
                np.sqrt(
                    max(
                        main_norm_squared,
                        0.0,
                    )
                )
            )

            aux_norm = float(
                np.sqrt(
                    max(
                        aux_norm_squared,
                        0.0,
                    )
                )
            )

            denominator = (
                main_norm
                * aux_norm
            )

            if denominator <= epsilon:
                zero_norm_count += 1
                continue

            cosine = float(
                dot_product
                / denominator
            )

            cosines.append(
                cosine
            )

            main_gradient_norms.append(
                main_norm
            )

            aux_gradient_norms.append(
                aux_norm
            )

        if not cosines:
            print(
                "GRAD_NEXT_DIAG | "
                f"year={year} | "
                f"n={len(X)} | "
                "valid=0"
            )
            continue

        cosine_array = np.asarray(
            cosines,
            dtype=np.float64,
        )

        print(
            "GRAD_NEXT_DIAG | "
            f"year={year} | "
            f"n={len(X)} | "
            f"valid={len(cosines)} | "
            f"zero_norm={zero_norm_count} | "
            f"cos_mean={np.mean(cosine_array):.6f} | "
            f"cos_median={np.median(cosine_array):.6f} | "
            f"cos_positive_rate="
            f"{np.mean(cosine_array > 0.0):.6f} | "
            f"cos_p25={np.percentile(cosine_array, 25):.6f} | "
            f"cos_p75={np.percentile(cosine_array, 75):.6f} | "
            f"main_grad_norm_mean="
            f"{np.mean(main_gradient_norms):.6f} | "
            f"aux_grad_norm_mean="
            f"{np.mean(aux_gradient_norms):.6f}"
        )

    if was_training:
        model.train()

def _print_one_step_metadata_diagnostics(
    *,
    model: nn.Module,
    stream: list[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ],
    device: torch.device,
    learning_rate: float = 1.0e-6,
    gradient_clip: float = 0.1,
) -> None:
    """
    Diagnostic only.

    For every temporal pair (t, t+1):

      1. start from the exact same source model;
      2. measure main loss on sample t+1;
      3. perform ONE Adam metadata update using sample t;
      4. measure main loss on sample t+1 again;
      5. rollback to the source adapter.

    No main-task label is ever used for adaptation.
    """

    probe_model = copy.deepcopy(
        model
    ).to(device)

    probe_model.eval()

    # Freeze everything.
    for parameter in probe_model.parameters():
        parameter.requires_grad_(False)

    # Metadata TTA updates only the online adapter.
    adapter_parameters = list(
        probe_model.online_adapter.parameters()
    )

    for parameter in adapter_parameters:
        parameter.requires_grad_(True)

    source_adapter_values = [
        parameter.detach().clone()
        for parameter in adapter_parameters
    ]

    for (
        year,
        X,
        y_main,
        y_aux,
    ) in stream:

        delta_losses: list[float] = []
        parameter_deltas: list[float] = []
        aux_losses: list[float] = []

        before_correct = 0
        after_correct = 0

        helpful_flips = 0
        harmful_flips = 0

        for index in range(
            len(X) - 1
        ):

            # ---------------------------------------------
            # Reset adapter exactly to source parameters.
            # ---------------------------------------------

            with torch.no_grad():

                for (
                    parameter,
                    source_value,
                ) in zip(
                    adapter_parameters,
                    source_adapter_values,
                ):

                    parameter.copy_(
                        source_value
                    )

            # Fresh Adam state for every pair.
            optimizer = torch.optim.Adam(
                adapter_parameters,
                lr=learning_rate,
            )

            # ---------------------------------------------
            # Current sample t: metadata available here.
            # ---------------------------------------------

            x_current = torch.as_tensor(
                X[index:index + 1],
                dtype=torch.float32,
                device=device,
            )

            y_aux_current = torch.as_tensor(
                y_aux[index:index + 1],
                dtype=torch.long,
                device=device,
            )

            # ---------------------------------------------
            # Next sample t+1: main label diagnostic only.
            # ---------------------------------------------

            x_next = torch.as_tensor(
                X[index + 1:index + 2],
                dtype=torch.float32,
                device=device,
            )

            y_main_next = torch.as_tensor(
                y_main[index + 1:index + 2],
                dtype=torch.long,
                device=device,
            )

            # ---------------------------------------------
            # Main performance BEFORE metadata update.
            # ---------------------------------------------

            with torch.no_grad():

                main_logits_before, _ = (
                    probe_model.forward_both(
                        x_next
                    )
                )

                main_loss_before = (
                    torch.nn.functional.cross_entropy(
                        main_logits_before,
                        y_main_next,
                    )
                )

                prediction_before = int(
                    main_logits_before
                    .argmax(dim=1)
                    .item()
                )

            # ---------------------------------------------
            # ONE metadata update on sample t.
            # ---------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            _, aux_logits = (
                probe_model.forward_both(
                    x_current
                )
            )

            aux_loss = (
                torch.nn.functional.cross_entropy(
                    aux_logits,
                    y_aux_current,
                )
            )

            aux_loss.backward()

            if gradient_clip > 0.0:

                torch.nn.utils.clip_grad_norm_(
                    adapter_parameters,
                    max_norm=gradient_clip,
                )

            optimizer.step()

            # ---------------------------------------------
            # Parameter displacement.
            # ---------------------------------------------

            squared_delta = 0.0

            for (
                parameter,
                source_value,
            ) in zip(
                adapter_parameters,
                source_adapter_values,
            ):

                difference = (
                    parameter.detach()
                    - source_value
                )

                squared_delta += float(
                    torch.sum(
                        difference.float() ** 2
                    ).item()
                )

            parameter_delta = float(
                np.sqrt(
                    max(
                        squared_delta,
                        0.0,
                    )
                )
            )

            # ---------------------------------------------
            # Main performance AFTER metadata update.
            # ---------------------------------------------

            with torch.no_grad():

                main_logits_after, _ = (
                    probe_model.forward_both(
                        x_next
                    )
                )

                main_loss_after = (
                    torch.nn.functional.cross_entropy(
                        main_logits_after,
                        y_main_next,
                    )
                )

                prediction_after = int(
                    main_logits_after
                    .argmax(dim=1)
                    .item()
                )

            target = int(
                y_main_next.item()
            )

            before_is_correct = (
                prediction_before == target
            )

            after_is_correct = (
                prediction_after == target
            )

            before_correct += int(
                before_is_correct
            )

            after_correct += int(
                after_is_correct
            )

            if (
                not before_is_correct
                and after_is_correct
            ):
                helpful_flips += 1

            if (
                before_is_correct
                and not after_is_correct
            ):
                harmful_flips += 1

            delta_losses.append(
                float(
                    main_loss_after.item()
                    - main_loss_before.item()
                )
            )

            parameter_deltas.append(
                parameter_delta
            )

            aux_losses.append(
                float(
                    aux_loss.detach().item()
                )
            )

        delta_array = np.asarray(
            delta_losses,
            dtype=np.float64,
        )

        n_pairs = len(
            delta_losses
        )

        print(
            "ONE_STEP_DIAG | "
            f"year={year} | "
            f"n_pairs={n_pairs} | "
            f"lr={learning_rate:.2e} | "
            f"clip={gradient_clip:.6g} | "
            f"delta_loss_mean="
            f"{np.mean(delta_array):.9f} | "
            f"delta_loss_median="
            f"{np.median(delta_array):.9f} | "
            f"loss_improve_rate="
            f"{np.mean(delta_array < 0.0):.6f} | "
            f"acc_before="
            f"{before_correct / n_pairs:.6f} | "
            f"acc_after="
            f"{after_correct / n_pairs:.6f} | "
            f"helpful_flips={helpful_flips} | "
            f"harmful_flips={harmful_flips} | "
            f"mean_aux_loss="
            f"{np.mean(aux_losses):.6f} | "
            f"mean_parameter_delta="
            f"{np.mean(parameter_deltas):.9f}"
        )

# ============================================================
# FROZEN REFERENCES
# ============================================================


def _frozen_accuracies(
    *,
    source_model: nn.Module,
    stream: list[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ],
    device: torch.device,
    family: str,
) -> dict[int, float]:

    values = {}

    for (
        year,
        X,
        y_main,
        _,
    ) in stream:

        record = (
            evaluate_frozen_year(
                model=source_model,
                X=X,
                y_main=y_main,
                year=year,
                method_name=(
                    f"{family}_frozen"
                ),
                device=device,
            )
        )

        values[
            int(year)
        ] = float(
            record.accuracy
        )

    return values


# ============================================================
# LOAD BEST BASELINE
# ============================================================


def _load_baseline_config(
    *,
    base_config: ExperimentConfig,
    baseline_best_path: Path,
) -> ExperimentConfig:

    result = load_best_result(
        baseline_best_path
    )

    overrides = result.get(
        "overrides",
        {},
    )

    if not isinstance(
        overrides,
        Mapping,
    ):
        raise ValueError(
            f"Invalid overrides in "
            f"{baseline_best_path}"
        )

    return apply_overrides(
        config=base_config,
        overrides=overrides,
    )


# ============================================================
# TTA PARAMETER INHERITANCE
# ============================================================


def _tta_parent_overrides(
    *,
    method_name: str,
    method_section: Mapping[str, Any],
    output_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:

    parent_name = method_section.get(
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

    mapping = method_section.get(
        "inherit_parameters",
        {},
    )

    if not isinstance(
        mapping,
        Mapping,
    ):
        raise ValueError(
            f"tta.methods.{method_name}."
            "inherit_parameters must be "
            "a mapping."
        )

    parent_best_path = (
        output_root
        / "tta"
        / parent_name
        / "best.yaml"
    )

    if not parent_best_path.is_file():
        raise FileNotFoundError(
            f"TTA method {method_name} "
            f"inherits from {parent_name}, "
            "but its result does not exist: "
            f"{parent_best_path}"
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
# ONE TTA METHOD
# ============================================================


def tune_tta_method(
    *,
    method_name: str,
    method_section: Mapping[str, Any],
    source_model: nn.Module,
    source_config: ExperimentConfig,
    stream: list[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ],
    frozen: Mapping[int, float],
    output_root: Path,
    device: torch.device,
) -> TTATuningResult:

    strategy = (
        build_search_strategy(
            str(
                method_section.get(
                    "search_strategy",
                    "grid",
                )
            )
        )
    )

    space = method_section.get(
        "space",
        {},
    )

    if not isinstance(
        space,
        Mapping,
    ):
        raise ValueError(
            f"{method_name}.space "
            "must be a mapping."
        )

    output_directory = (
        ensure_directory(
            output_root
            / "tta"
            / method_name
        )
    )

    (
        inherited_overrides,
        inheritance_info,
    ) = _tta_parent_overrides(
        method_name=method_name,
        method_section=method_section,
        output_root=output_root,
    )

    if inherited_overrides:

        print()
        print(
            f"{method_name} inherited "
            "TTA parameters:"
        )

        for path, value in (
            inherited_overrides.items()
        ):
            print(
                f"  {path} = {value}"
            )

    rows: list[
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

    method_class = (
        get_method_class(
            method_name
        )
    )

    for trial in strategy.generate(
        space
    ):

        print()
        print(
            "=" * 80
        )
        print(
            f"TTA TUNING | "
            f"{method_name} | "
            f"trial={trial.trial_id}"
        )
        print(
            trial.parameters
        )
        print(
            "=" * 80
        )

        complete_overrides = (
            merge_overrides(
                inherited_overrides,
                trial.overrides,
            )
        )

        trial_config = (
            apply_overrides(
                config=source_config,
                overrides=(
                    complete_overrides
                ),
            )
        )

        set_seed(
            trial_config.seed
        )

        method_config = (
            trial_config.section(
                method_name
            )
        )

        print("DEBUG_METADATA | before_method_construction", flush=True)

        # IMPORTANT:
        #
        # Every TTA trial starts from exactly the SAME source
        # checkpoint.
        #
        # Temporal Gradient / Consensus inherit only TTA
        # hyperparameters from Metadata, NOT Metadata-adapted
        # weights.
        method = method_class(
            source_model=copy.deepcopy(
                source_model
            ),
            config=method_config,
            device=device,
        )

        print("DEBUG_METADATA | after_method_construction", flush=True)

        print("DEBUG_METADATA | before_evaluation", flush=True)

        # Continuous temporal stream.
        #
        # State is never reset between pseudo-OOD years.
        evaluation = (
            evaluate_tta_stream(
                method=method,
                years=stream,
                reset_each_year=False,
            )
        )

        print("DEBUG_METADATA | after_evaluation", flush=True)

        tta_by_year = {
            int(record.year):
                float(
                    record.accuracy
                )
            for record
            in evaluation.records
        }

        deltas = {
            year:
                tta_by_year[year]
                - float(
                    frozen[year]
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

        std = float(
            pstdev(
                delta_values
            )
            if len(
                delta_values
            ) > 1
            else 0.0
        )

        worst = float(
            min(
                delta_values
            )
        )

        n_improved = int(
            sum(
                value > 0.0
                for value
                in delta_values
            )
        )

        row: dict[
            str,
            Any,
        ] = {
            "trial_id":
                trial.trial_id,
            "mean_delta_accuracy":
                score,
            "std_delta_accuracy":
                std,
            "worst_delta_accuracy":
                worst,
            "n_improved_years":
                n_improved,
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

        if score > best_score:

            best_score = score

            # Only parameters searched in THIS stage.
            best_parameters = dict(
                trial.parameters
            )

            # Complete config for THIS method, including the
            # inherited common values.
            best_overrides = dict(
                complete_overrides
            )

        save_trial_rows(
            output_directory
            / "trials.csv",
            rows,
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
            "stream_years": [
                year
                for (
                    year,
                    _,
                    _,
                    _,
                )
                in stream
            ],
            "reset_each_year":
                False,
            **inheritance_info,
        },
    )

    return TTATuningResult(
        method_name=method_name,
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
# FULL STAGED TTA PIPELINE
# ============================================================


def tune_tta(
    *,
    base_config: ExperimentConfig,
    tuning_config: Mapping[str, Any],
    output_root: Path,
) -> dict[
    str,
    TTATuningResult,
]:

    tta_config = tuning_config.get(
        "tta",
        {},
    )

    if not isinstance(
        tta_config,
        Mapping,
    ):
        raise ValueError(
            "tta tuning section must "
            "be a mapping."
        )

    protocol = base_config.section(
        "protocol"
    )

    source_start = int(
        protocol[
            "source_start_year"
        ]
    )

    source_end = int(
        protocol[
            "source_end_year"
        ]
    )

    stream_start = int(
        tta_config[
            "stream_start_year"
        ]
    )

    if not (
        source_start
        < stream_start
        <= source_end
    ):
        raise ValueError(
            "tta.stream_start_year must "
            "be after source_start_year and "
            "<= source_end_year."
        )

    supervised_years = list(
        range(
            source_start,
            stream_start,
        )
    )

    stream_years = list(
        range(
            stream_start,
            source_end + 1,
        )
    )

    print()
    print(
        "=" * 80
    )
    print(
        "TTA TUNING TEMPORAL SPLIT"
    )
    print(
        f"Pseudo-source years: "
        f"{supervised_years}"
    )
    print(
        f"Pseudo-OOD stream years: "
        f"{stream_years}"
    )
    print(
        "=" * 80
    )

    bundle = build_data_bundle(
        base_config
    )

    device = _device()

    baseline_root = (
        output_root
        / "baselines"
    )

    single_best = (
        baseline_root
        / "single_head"
        / "best.yaml"
    )

    if not single_best.is_file():
        raise FileNotFoundError(
            "TTA tuning requires the tuned "
            "single-head baseline first: "
            f"{single_best}"
        )

    single_best_result = load_best_result(
        single_best
    )

    baseline_years = (
        single_best_result
        .get("extra", {})
        .get("years", [])
    )

    if not baseline_years:
        raise ValueError(
            "Single-head best.yaml does not record tuning years."
        )

    leaking_years = [
        int(year)
        for year in baseline_years
        if int(year) >= stream_start
    ]

    if leaking_years:
        raise ValueError(
            "Temporal leakage: single-head tuning used years "
            f"inside the pseudo-OOD TTA stream: {leaking_years}"
        )

    single_config = (
        _load_baseline_config(
            base_config=base_config,
            baseline_best_path=(
                single_best
            ),
        )
    )

    # The aux-only double is derived directly from the tuned
    # single-head source model. No jointly-trained double-head
    # baseline is required anymore.
    double_config = apply_overrides(
        config=single_config,
        overrides={
            "model.double_head.dropout":
                single_config.get(
                    "model",
                    "single_head",
                    "dropout",
                ),
            "model.double_head.shared_hidden_dim":
                single_config.get(
                    "model",
                    "single_head",
                    "shared_hidden_dim",
                ),
        },
    )

    # ========================================================
    # TRAIN PSEUDO-SOURCE SINGLE
    # ========================================================

    set_seed(
        base_config.seed
    )

    source_single = (
        _train_source_model(
            family="single_head",
            config=single_config,
            bundle=bundle,
            source_years=(
                supervised_years
            ),
            device=device,
        )
    )

    # ========================================================
    # BUILD PSEUDO-SOURCE DOUBLE FROM TRAINED SINGLE
    # ========================================================

    source_double = (
        _train_aux_only_double_from_single(
            source_single=source_single,
            config=double_config,
            bundle=bundle,
            source_years=(
                supervised_years
            ),
            device=device,
        )
    )

    # ========================================================
    # SAME TEMPORAL STREAM
    # ========================================================

    single_stream = (
        _build_pseudo_stream(
            bundle=bundle,
            config=single_config,
            years=stream_years,
        )
    )

    double_stream = (
        _build_pseudo_stream(
            bundle=bundle,
            config=double_config,
            years=stream_years,
        )
    )

    # ========================================================
    # ONE-STEP METADATA DIAGNOSTIC
    # ========================================================

    # _print_one_step_metadata_diagnostics(
    #     model=source_double,
    #     stream=double_stream,
    #     device=device,
    #     learning_rate=3.0e-4,
    #     gradient_clip=0.1,
    # )

    # ========================================================
    # FROZEN REFERENCES
    # ========================================================

    frozen_single = (
        _frozen_accuracies(
            source_model=source_single,
            stream=single_stream,
            device=device,
            family="single_head",
        )
    )

    frozen_double = (
        _frozen_accuracies(
            source_model=source_double,
            stream=double_stream,
            device=device,
            family="double_head",
        )
    )

    # ========================================================
    # SINGLE -> AUX-ONLY DOUBLE MAIN-PATH SANITY CHECK
    # ========================================================

    for year in stream_years:

        single_accuracy = float(
            frozen_single[year]
        )

        double_accuracy = float(
            frozen_double[year]
        )

        difference = (
            double_accuracy
            - single_accuracy
        )

        print(
            "AUX_ONLY_MAIN_SANITY | "
            f"year={year} | "
            f"single={single_accuracy:.12f} | "
            f"double={double_accuracy:.12f} | "
            f"diff={difference:.12e}"
        )

        if abs(difference) > 1.0e-12:
            raise RuntimeError(
                "Aux-only double changed main-task "
                f"accuracy in year {year}: "
                f"single={single_accuracy}, "
                f"double={double_accuracy}."
            )

    methods_config = (
        tta_config.get(
            "methods",
            {},
        )
    )

    if not isinstance(
        methods_config,
        Mapping,
    ):
        raise ValueError(
            "tta.methods must be "
            "a mapping."
        )

    results: dict[
        str,
        TTATuningResult,
    ] = {}

    # --------------------------------------------------------
    # Dependency order is EXPLICIT.
    #
    # Metadata must run before Temporal Gradient / Consensus.
    # TENT is independent but uses the tuned single baseline.
    # --------------------------------------------------------

    ordered_methods = (
        "metadata",
        "temporal_gradient",
        "experimental",
        "consensus",
        "tent",
    )

    for method_name in (
        ordered_methods
    ):

        method_section = (
            methods_config.get(
                method_name
            )
        )

        if method_section is None:
            continue

        if not isinstance(
            method_section,
            Mapping,
        ):
            raise ValueError(
                f"tta.methods.{method_name} "
                "must be a mapping."
            )

        if not bool(
            method_section.get(
                "enabled",
                True,
            )
        ):
            continue

        family = _method_family(
            method_name
        )

        if family == "single_head":

            source_model = (
                source_single
            )

            source_config = (
                single_config
            )

            stream = (
                single_stream
            )

            frozen = (
                frozen_single
            )

        else:

            source_model = (
                source_double
            )

            source_config = (
                double_config
            )

            stream = (
                double_stream
            )

            frozen = (
                frozen_double
            )

        results[
            method_name
        ] = tune_tta_method(
            method_name=(
                method_name
            ),
            method_section=(
                method_section
            ),
            source_model=(
                source_model
            ),
            source_config=(
                source_config
            ),
            stream=stream,
            frozen=frozen,
            output_root=(
                output_root
            ),
            device=device,
        )

    return results