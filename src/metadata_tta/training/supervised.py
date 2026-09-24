from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from metadata_tta.config import (
    ExperimentConfig,
)

from metadata_tta.models import (
    DoubleHeadClassifier,
    SingleHeadClassifier,
)


# ============================================================
# TRAINING RESULT
# ============================================================

@dataclass
class TrainingResult:
    """
    Result of one supervised training stage.
    """

    model: nn.Module

    optimizer: Optimizer

    # Number of epochs actually executed.
    epochs: int

    learning_rate: float

    mean_epoch_losses: list[float]

    validation_losses: list[float]

    best_epoch: int

    best_validation_loss: float

    stopped_early: bool

# ============================================================
# TEMPORAL TRAINING SCHEDULE
# ============================================================

@dataclass(frozen=True)
class TrainingSchedule:
    """
    Hyperparameters used for one temporal supervised stage.
    """

    learning_rate: float

    epochs: int

    batch_size: int

    weight_decay: float

    reset_optimizer: bool

    initialized_from_previous: bool

    early_stopping_enabled: bool

    early_stopping_patience: int

    early_stopping_min_delta: float

def _merged_training_section(
    config: ExperimentConfig,
    model_kind: str | None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:

    training = config.section(
        "training"
    )

    base = dict(
        training[
            "base"
        ]
    )

    temporal = dict(
        training[
            "temporal"
        ]
    )

    if model_kind is None:
        return (
            base,
            temporal,
        )

    head_section = training.get(
        model_kind,
        {}
    )

    if not isinstance(
        head_section,
        dict,
    ):
        return (
            base,
            temporal,
        )

    # Optional per-head base-training overrides.
    for key in (
        "learning_rate",
        "weight_decay",
        "epochs",
        "batch_size",
    ):
        if key in head_section:
            base[
                key
            ] = head_section[
                key
            ]

    head_temporal = (
        head_section.get(
            "temporal",
            {},
        )
    )

    if isinstance(
        head_temporal,
        dict,
    ):
        temporal.update(
            head_temporal
        )

    return (
        base,
        temporal,
    )

def build_training_schedule(
    config: ExperimentConfig,
    initialized_from_previous: bool,
    model_kind: str | None = None,
) -> TrainingSchedule:
    """
    Determine supervised training hyperparameters.

    model_kind can be:
        None
        "single_head"
        "double_head"

    The first supervised stage uses training.base.

    Later yearly stages use training.temporal when enabled.

    Early stopping settings follow the same rule:
        training.base.early_stopping
        training.temporal.early_stopping
    """

    (
        base,
        temporal,
    ) = _merged_training_section(
        config=config,
        model_kind=model_kind,
    )

    if (
        initialized_from_previous
        and bool(
            temporal[
                "enabled"
            ]
        )
    ):

        learning_rate = float(
            temporal[
                "learning_rate"
            ]
        )

        epochs = int(
            temporal[
                "epochs"
            ]
        )

        reset_optimizer = bool(
            temporal[
                "reset_optimizer_each_year"
            ]
        )

        early_stopping = temporal.get(
            "early_stopping",
            {},
        )

    else:

        learning_rate = float(
            base[
                "learning_rate"
            ]
        )

        epochs = int(
            base[
                "epochs"
            ]
        )

        reset_optimizer = True

        early_stopping = base.get(
            "early_stopping",
            {},
        )

    if not isinstance(
        early_stopping,
        dict,
    ):
        early_stopping = {}

    early_stopping_enabled = bool(
        early_stopping.get(
            "enabled",
            False,
        )
    )

    early_stopping_patience = int(
        early_stopping.get(
            "patience",
            1,
        )
    )

    early_stopping_min_delta = float(
        early_stopping.get(
            "min_delta",
            0.0,
        )
    )

    return TrainingSchedule(
        learning_rate=learning_rate,

        epochs=epochs,

        batch_size=int(
            base[
                "batch_size"
            ]
        ),

        weight_decay=float(
            base[
                "weight_decay"
            ]
        ),

        reset_optimizer=(
            reset_optimizer
        ),

        initialized_from_previous=bool(
            initialized_from_previous
        ),

        early_stopping_enabled=(
            early_stopping_enabled
        ),

        early_stopping_patience=(
            early_stopping_patience
        ),

        early_stopping_min_delta=(
            early_stopping_min_delta
        ),
    )

def _merged_model_config(
    config: ExperimentConfig,
    model_kind: str,
) -> dict[str, Any]:

    model_config = (
        config.section(
            "model"
        )
    )

    head_overrides = (
        model_config.get(
            model_kind,
            {},
        )
    )

    if isinstance(
        head_overrides,
        dict,
    ):
        model_config.update(
            head_overrides
        )

    return model_config

# ============================================================
# MODEL CREATION
# ============================================================

def create_single_head_model(
    input_dim: int,
    config: ExperimentConfig,
) -> SingleHeadClassifier:
    """
    Create a new SingleHeadClassifier from configuration.
    """

    dataset = config.section(
        "dataset"
    )

    model_config = _merged_model_config(
        config=config,
        model_kind="single_head",
    )

    return SingleHeadClassifier(
        input_dim=int(
            input_dim
        ),
        n_classes_main=int(
            dataset[
                "n_classes_main"
            ]
        ),
        shared_hidden_dim=int(
            model_config[
                "shared_hidden_dim"
            ]
        ),
        bottleneck_dim=int(
            model_config[
                "adapter_bottleneck_dim"
            ]
        ),
        dropout=float(
            model_config[
                "dropout"
            ]
        ),
        use_shared_trunk=bool(
            model_config[
                "use_shared_trunk"
            ]
        ),
    )


def create_double_head_model(
    input_dim: int,
    config: ExperimentConfig,
) -> DoubleHeadClassifier:
    """
    Create a new DoubleHeadClassifier from configuration.
    """

    dataset = config.section(
        "dataset"
    )

    model_config = _merged_model_config(
        config=config,
        model_kind="double_head",
    )

    return DoubleHeadClassifier(
        input_dim=int(
            input_dim
        ),
        n_classes_main=int(
            dataset[
                "n_classes_main"
            ]
        ),
        n_classes_aux=int(
            dataset[
                "n_classes_aux"
            ]
        ),
        shared_hidden_dim=int(
            model_config[
                "shared_hidden_dim"
            ]
        ),
        bottleneck_dim=int(
            model_config[
                "adapter_bottleneck_dim"
            ]
        ),
        dropout=float(
            model_config[
                "dropout"
            ]
        ),
        use_shared_trunk=bool(
            model_config[
                "use_shared_trunk"
            ]
        ),
    )

def initialize_double_from_single(
    *,
    single_model: SingleHeadClassifier,
    double_model: DoubleHeadClassifier,
) -> DoubleHeadClassifier:
    """
    Copy the complete main-task path from an already trained
    single-head model into a double-head model.

    The auxiliary head remains at its own initialization.
    """

    single_state = single_model.state_dict()
    double_state = double_model.state_dict()

    transferable_state = {}

    for key, value in single_state.items():

        if key not in double_state:
            raise RuntimeError(
                "Single-to-double initialization failed: "
                f"missing target key {key!r}."
            )

        if (
            double_state[key].shape
            != value.shape
        ):
            raise RuntimeError(
                "Single-to-double initialization failed: "
                f"shape mismatch for {key!r}: "
                f"{tuple(value.shape)} vs "
                f"{tuple(double_state[key].shape)}."
            )

        transferable_state[key] = (
            value.detach().clone()
        )

    incompatible = (
        double_model.load_state_dict(
            transferable_state,
            strict=False,
        )
    )

    expected_missing = {
        "aux_head.weight",
        "aux_head.bias",
    }

    if (
        set(incompatible.missing_keys)
        != expected_missing
    ):
        raise RuntimeError(
            "Unexpected missing keys during "
            "single-to-double initialization: "
            f"{incompatible.missing_keys}."
        )

    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected keys during "
            "single-to-double initialization: "
            f"{incompatible.unexpected_keys}."
        )

    return double_model

def train_aux_head_only(
    *,
    model: DoubleHeadClassifier,
    X: np.ndarray,
    y_aux: np.ndarray,
    X_validation: np.ndarray,
    y_aux_validation: np.ndarray,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    min_delta: float,
    device: torch.device,
    log_prefix: str | None = None,
) -> DoubleHeadClassifier:
    """
    Train ONLY the auxiliary head.

    The complete main-task path remains frozen and in eval mode,
    so its parameters, BatchNorm statistics and dropout behaviour
    cannot change.
    """

    X = np.asarray(
        X,
        dtype=np.float32,
    )
    y_aux = np.asarray(
        y_aux,
        dtype=np.int64,
    )

    X_validation = np.asarray(
        X_validation,
        dtype=np.float32,
    )
    y_aux_validation = np.asarray(
        y_aux_validation,
        dtype=np.int64,
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Freeze the complete model, then enable aux head only.
    # --------------------------------------------------------

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for parameter in model.aux_head.parameters():
        parameter.requires_grad_(True)

    # Keep backbone / BatchNorm / Dropout deterministic.
    model.eval()
    model.aux_head.train()

    optimizer = Adam(
        model.aux_head.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(
        torch.as_tensor(
            X,
            dtype=torch.float32,
        ),
        torch.as_tensor(
            y_aux,
            dtype=torch.long,
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )

    validation_dataset = TensorDataset(
        torch.as_tensor(
            X_validation,
            dtype=torch.float32,
        ),
        torch.as_tensor(
            y_aux_validation,
            dtype=torch.long,
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
    )

    best_validation_loss = float("inf")
    best_aux_state = copy.deepcopy(
        model.aux_head.state_dict()
    )

    epochs_without_improvement = 0

    prefix = (
        f"[{log_prefix}] "
        if log_prefix
        else ""
    )

    for epoch_index in range(int(epochs)):

        epoch = epoch_index + 1

        model.eval()
        model.aux_head.train()

        total_train_loss = 0.0
        total_train_samples = 0

        for (
            batch_X,
            batch_aux,
        ) in train_loader:

            batch_X = batch_X.to(device)
            batch_aux = batch_aux.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            # Frozen deterministic representation.
            with torch.no_grad():
                shared = (
                    model.extract_shared_features(
                        batch_X
                    )
                )

            aux_logits = model.aux_head(
                shared
            )

            loss = criterion(
                aux_logits,
                batch_aux,
            )

            loss.backward()
            optimizer.step()

            batch_n = int(
                len(batch_X)
            )

            total_train_loss += (
                float(loss.detach().item())
                * batch_n
            )

            total_train_samples += batch_n

        train_loss = (
            total_train_loss
            / max(
                total_train_samples,
                1,
            )
        )

        # ----------------------------------------------------
        # AUX VALIDATION
        # ----------------------------------------------------

        model.eval()

        total_validation_loss = 0.0
        total_validation_samples = 0

        with torch.no_grad():

            for (
                batch_X,
                batch_aux,
            ) in validation_loader:

                batch_X = batch_X.to(device)
                batch_aux = batch_aux.to(device)

                shared = (
                    model.extract_shared_features(
                        batch_X
                    )
                )

                aux_logits = model.aux_head(
                    shared
                )

                loss = criterion(
                    aux_logits,
                    batch_aux,
                )

                batch_n = int(
                    len(batch_X)
                )

                total_validation_loss += (
                    float(loss.item())
                    * batch_n
                )

                total_validation_samples += batch_n

        validation_loss = (
            total_validation_loss
            / max(
                total_validation_samples,
                1,
            )
        )

        improved = (
            validation_loss
            < (
                best_validation_loss
                - float(min_delta)
            )
        )

        if improved:

            best_validation_loss = (
                validation_loss
            )

            best_aux_state = copy.deepcopy(
                model.aux_head.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        print(
            f"{prefix}"
            f"aux-only epoch {epoch}/{epochs} | "
            f"train_aux_loss={train_loss:.6f} | "
            f"val_aux_loss={validation_loss:.6f} | "
            f"best_val_aux_loss="
            f"{best_validation_loss:.6f}"
        )

        if (
            epochs_without_improvement
            >= int(patience)
        ):

            print(
                f"{prefix}"
                f"AUX-ONLY EARLY STOP "
                f"at epoch {epoch}"
            )

            break

    model.aux_head.load_state_dict(
        best_aux_state
    )

    # Restore gradients for later TTA configuration.
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    model.eval()

    return model

# ============================================================
# OPTIMIZER
# ============================================================

def create_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> Optimizer:
    """
    Create the supervised optimizer.

    Adam is used for the supervised models.
    """

    return Adam(
        model.parameters(),
        lr=float(
            learning_rate
        ),
        weight_decay=float(
            weight_decay
        ),
    )


def set_optimizer_learning_rate(
    optimizer: Optimizer,
    learning_rate: float,
) -> None:
    """
    Change the learning rate of an existing optimizer.
    """

    learning_rate = float(
        learning_rate
    )

    for parameter_group in (
        optimizer.param_groups
    ):
        parameter_group[
            "lr"
        ] = learning_rate


# ============================================================
# INPUT VALIDATION
# ============================================================

def _validate_training_arrays(
    X: np.ndarray,
    y_main: np.ndarray,
    y_aux: np.ndarray | None = None,
) -> None:

    if X.ndim != 2:
        raise ValueError(
            f"X must be 2-D, got {X.shape}."
        )

    if y_main.ndim != 1:
        raise ValueError(
            "y_main must be 1-D."
        )

    if len(X) != len(y_main):
        raise ValueError(
            "X and y_main must contain "
            "the same number of samples."
        )

    if y_aux is not None:

        if y_aux.ndim != 1:
            raise ValueError(
                "y_aux must be 1-D."
            )

        if len(X) != len(y_aux):
            raise ValueError(
                "X and y_aux must contain "
                "the same number of samples."
            )

    if len(X) == 0:
        raise ValueError(
            "Cannot train on an empty dataset."
        )

def _should_drop_last(
    n_samples: int,
    batch_size: int,
) -> bool:
    """
    Drop the final batch only when it would contain
    exactly one sample.

    This is required by BatchNorm1d during supervised
    training, while avoiding unnecessary sample loss.
    """

    n_samples = int(n_samples)
    batch_size = int(batch_size)

    return (
        n_samples > 1
        and batch_size > 1
        and n_samples % batch_size == 1
    )

# ============================================================
# DATA LOADERS
# ============================================================

def _single_head_loader(
    X: np.ndarray,
    y_main: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:

    dataset = TensorDataset(
        torch.as_tensor(
            X,
            dtype=torch.float32,
        ),
        torch.as_tensor(
            y_main,
            dtype=torch.long,
        ),
    )

    return DataLoader(
        dataset,
        batch_size=int(
            batch_size
        ),
        shuffle=bool(
            shuffle
        ),
        drop_last=(
            _should_drop_last(
                n_samples=len(dataset),
                batch_size=batch_size,
            )
            if shuffle
            else False
        ),
    )

def _double_head_loader(
    X: np.ndarray,
    y_main: np.ndarray,
    y_aux: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:

    dataset = TensorDataset(
        torch.as_tensor(
            X,
            dtype=torch.float32,
        ),
        torch.as_tensor(
            y_main,
            dtype=torch.long,
        ),
        torch.as_tensor(
            y_aux,
            dtype=torch.long,
        ),
    )

    return DataLoader(
        dataset,
        batch_size=int(
            batch_size
        ),
        shuffle=bool(
            shuffle
        ),
        drop_last=(
            _should_drop_last(
                n_samples=len(dataset),
                batch_size=batch_size,
            )
            if shuffle
            else False
        ),
    )

def _single_head_validation_loss(
    model: SingleHeadClassifier,
    X: np.ndarray,
    y_main: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:

    loader = _single_head_loader(
        X=X,
        y_main=y_main,
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()

    was_training = model.training

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for (
            batch_X,
            batch_y,
        ) in loader:

            batch_X = batch_X.to(
                device
            )

            batch_y = batch_y.to(
                device
            )

            logits = model(
                batch_X
            )

            loss = criterion(
                logits,
                batch_y,
            )

            batch_n = int(
                len(
                    batch_X
                )
            )

            total_loss += (
                float(
                    loss.item()
                )
                * batch_n
            )

            total_samples += (
                batch_n
            )

    if was_training:
        model.train()

    return (
        total_loss
        / max(
            total_samples,
            1,
        )
    )


def _double_head_validation_loss(
    model: DoubleHeadClassifier,
    X: np.ndarray,
    y_main: np.ndarray,
    y_aux: np.ndarray,
    batch_size: int,
    aux_loss_weight: float,
    device: torch.device,
) -> float:
    """
    Validation objective is the same objective used for training:

        CE_main + aux_loss_weight * CE_aux
    """

    loader = _double_head_loader(
        X=X,
        y_main=y_main,
        y_aux=y_aux,
        batch_size=batch_size,
        shuffle=False,
    )

    main_criterion = (
        nn.CrossEntropyLoss()
    )

    aux_criterion = (
        nn.CrossEntropyLoss()
    )

    was_training = model.training

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for (
            batch_X,
            batch_main,
            batch_aux,
        ) in loader:

            batch_X = batch_X.to(
                device
            )

            batch_main = batch_main.to(
                device
            )

            batch_aux = batch_aux.to(
                device
            )

            (
                main_logits,
                aux_logits,
            ) = model.forward_both(
                batch_X
            )

            main_loss = (
                main_criterion(
                    main_logits,
                    batch_main,
                )
            )

            aux_loss = (
                aux_criterion(
                    aux_logits,
                    batch_aux,
                )
            )

            loss = (
                main_loss
                + float(
                    aux_loss_weight
                )
                * aux_loss
            )

            batch_n = int(
                len(
                    batch_X
                )
            )

            total_loss += (
                float(
                    loss.item()
                )
                * batch_n
            )

            total_samples += (
                batch_n
            )

    if was_training:
        model.train()

    return (
        total_loss
        / max(
            total_samples,
            1,
        )
    )

# ============================================================
# SINGLE HEAD TRAINING
# ============================================================

def train_single_head(
    model: SingleHeadClassifier,
    X: np.ndarray,
    y_main: np.ndarray,
    schedule: TrainingSchedule,
    device: torch.device,
    optimizer: Optimizer | None = None,
    X_validation: np.ndarray | None = None,
    y_main_validation: np.ndarray | None = None,
    log_prefix: str | None = None,
) -> TrainingResult:
    """
    Train or temporally fine-tune a Single Head model.

    When early stopping is enabled, the model and optimizer
    state corresponding to the best validation loss are restored.
    """

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y_main = np.asarray(
        y_main,
        dtype=np.int64,
    )

    _validate_training_arrays(
        X=X,
        y_main=y_main,
    )

    if (
        X_validation is None
    ) != (
        y_main_validation is None
    ):
        raise ValueError(
            "X_validation and y_main_validation "
            "must either both be provided or both be None."
        )

    has_validation = (
        X_validation is not None
    )

    if (
        schedule.early_stopping_enabled
        and not has_validation
    ):
        raise ValueError(
            "Early stopping is enabled but no "
            "validation data was provided."
        )

    if has_validation:

        X_validation = np.asarray(
            X_validation,
            dtype=np.float32,
        )

        y_main_validation = np.asarray(
            y_main_validation,
            dtype=np.int64,
        )

        _validate_training_arrays(
            X=X_validation,
            y_main=y_main_validation,
        )

    model = model.to(
        device
    )

    if (
        optimizer is None
        or schedule.reset_optimizer
    ):
        optimizer = create_optimizer(
            model=model,
            learning_rate=(
                schedule.learning_rate
            ),
            weight_decay=(
                schedule.weight_decay
            ),
        )

    else:
        set_optimizer_learning_rate(
            optimizer=optimizer,
            learning_rate=(
                schedule.learning_rate
            ),
        )

    criterion = nn.CrossEntropyLoss()

    loader = _single_head_loader(
        X=X,
        y_main=y_main,
        batch_size=(
            schedule.batch_size
        ),
        shuffle=True,
    )

    mean_epoch_losses: list[
        float
    ] = []

    validation_losses: list[
        float
    ] = []

    best_validation_loss = float(
        "inf"
    )

    best_epoch = 0

    best_model_state = None
    best_optimizer_state = None

    epochs_without_improvement = 0

    stopped_early = False

    prefix = (
        f"[{log_prefix}] "
        if log_prefix
        else ""
    )

    model.train()

    for epoch_index in range(
        schedule.epochs
    ):

        epoch = (
            epoch_index + 1
        )

        total_loss = 0.0
        total_samples = 0

        for (
            batch_X,
            batch_y,
        ) in loader:

            batch_X = batch_X.to(
                device
            )

            batch_y = batch_y.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                batch_X
            )

            loss = criterion(
                logits,
                batch_y,
            )

            loss.backward()

            optimizer.step()

            batch_n = int(
                len(
                    batch_X
                )
            )

            total_loss += (
                float(
                    loss.detach().item()
                )
                * batch_n
            )

            total_samples += (
                batch_n
            )

        train_loss = (
            total_loss
            / max(
                total_samples,
                1,
            )
        )

        mean_epoch_losses.append(
            train_loss
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        if has_validation:

            val_loss = (
                _single_head_validation_loss(
                    model=model,
                    X=X_validation,
                    y_main=(
                        y_main_validation
                    ),
                    batch_size=(
                        schedule.batch_size
                    ),
                    device=device,
                )
            )

            validation_losses.append(
                val_loss
            )

            if schedule.early_stopping_enabled:

                improved = (
                    val_loss
                    < (
                        best_validation_loss
                        - schedule.early_stopping_min_delta
                    )
                )

            else:

                improved = (
                    val_loss
                    < best_validation_loss
                )

            if improved:

                best_validation_loss = (
                    val_loss
                )

                best_epoch = epoch

                best_model_state = (
                    copy.deepcopy(
                        model.state_dict()
                    )
                )

                best_optimizer_state = (
                    copy.deepcopy(
                        optimizer.state_dict()
                    )
                )

                epochs_without_improvement = 0

            elif schedule.early_stopping_enabled:

                epochs_without_improvement += 1

            if schedule.early_stopping_enabled:

                print(
                    f"{prefix}"
                    f"epoch {epoch}/{schedule.epochs} | "
                    f"train_loss={train_loss:.6f} | "
                    f"val_loss={val_loss:.6f} | "
                    f"best_val_loss="
                    f"{best_validation_loss:.6f} | "
                    f"best_epoch={best_epoch} | "
                    f"patience="
                    f"{epochs_without_improvement}/"
                    f"{schedule.early_stopping_patience}"
                )

            else:

                print(
                    f"{prefix}"
                    f"epoch {epoch}/{schedule.epochs} | "
                    f"train_loss={train_loss:.6f} | "
                    f"val_loss={val_loss:.6f}"
                )

            # =================================================
            # EARLY STOP
            # =================================================

            if (
                schedule.early_stopping_enabled
                and epochs_without_improvement
                >= schedule.early_stopping_patience
            ):

                stopped_early = True

                print(
                    f"{prefix}"
                    f"EARLY STOP at epoch {epoch} | "
                    f"best_epoch={best_epoch} | "
                    f"best_val_loss="
                    f"{best_validation_loss:.6f}"
                )

                break

        else:

            print(
                f"{prefix}"
                f"epoch {epoch}/{schedule.epochs} | "
                f"train_loss={train_loss:.6f}"
            )

    executed_epochs = len(
        mean_epoch_losses
    )

    # ========================================================
    # RESTORE BEST CHECKPOINT
    # ========================================================

    if (
        schedule.early_stopping_enabled
        and best_model_state is not None
    ):

        model.load_state_dict(
            best_model_state
        )

        optimizer.load_state_dict(
            best_optimizer_state
        )

    # ========================================================
    # FINAL LOG
    # ========================================================

    if has_validation:

        print(
            f"{prefix}"
            f"TRAINING FINISHED | "
            f"executed_epochs={executed_epochs} | "
            f"best_epoch={best_epoch} | "
            f"best_val_loss="
            f"{best_validation_loss:.6f} | "
            f"stopped_early={stopped_early}"
        )

    else:

        print(
            f"{prefix}"
            f"TRAINING FINISHED | "
            f"executed_epochs={executed_epochs}"
        )

    return TrainingResult(
        model=model,

        optimizer=optimizer,

        epochs=executed_epochs,

        learning_rate=float(
            schedule.learning_rate
        ),

        mean_epoch_losses=(
            mean_epoch_losses
        ),

        validation_losses=(
            validation_losses
        ),

        best_epoch=best_epoch,

        best_validation_loss=(
            best_validation_loss
        ),

        stopped_early=(
            stopped_early
        ),
    )

# ============================================================
# DOUBLE HEAD TRAINING
# ============================================================

def train_double_head(
    model: DoubleHeadClassifier,
    X: np.ndarray,
    y_main: np.ndarray,
    y_aux: np.ndarray,
    schedule: TrainingSchedule,
    aux_loss_weight: float,
    device: torch.device,
    optimizer: Optimizer | None = None,
    X_validation: np.ndarray | None = None,
    y_main_validation: np.ndarray | None = None,
    y_aux_validation: np.ndarray | None = None,
    log_prefix: str | None = None,
) -> TrainingResult:
    """
    Train or temporally fine-tune a Double Head model.

    Training and validation objective:

        L = L_main + lambda_aux * L_aux

    When early stopping is enabled, the model and optimizer
    state corresponding to the best validation loss are restored.
    """

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y_main = np.asarray(
        y_main,
        dtype=np.int64,
    )

    y_aux = np.asarray(
        y_aux,
        dtype=np.int64,
    )

    _validate_training_arrays(
        X=X,
        y_main=y_main,
        y_aux=y_aux,
    )

    validation_values = (
        X_validation,
        y_main_validation,
        y_aux_validation,
    )

    has_any_validation = any(
        value is not None
        for value in validation_values
    )

    has_all_validation = all(
        value is not None
        for value in validation_values
    )

    if (
        has_any_validation
        and not has_all_validation
    ):
        raise ValueError(
            "X_validation, y_main_validation and "
            "y_aux_validation must either all be provided "
            "or all be None."
        )

    has_validation = (
        has_all_validation
    )

    if (
        schedule.early_stopping_enabled
        and not has_validation
    ):
        raise ValueError(
            "Early stopping is enabled but no "
            "validation data was provided."
        )

    if has_validation:

        X_validation = np.asarray(
            X_validation,
            dtype=np.float32,
        )

        y_main_validation = np.asarray(
            y_main_validation,
            dtype=np.int64,
        )

        y_aux_validation = np.asarray(
            y_aux_validation,
            dtype=np.int64,
        )

        _validate_training_arrays(
            X=X_validation,
            y_main=y_main_validation,
            y_aux=y_aux_validation,
        )

    aux_loss_weight = float(
        aux_loss_weight
    )

    if aux_loss_weight < 0.0:
        raise ValueError(
            "aux_loss_weight must be >= 0."
        )

    model = model.to(
        device
    )

    if (
        optimizer is None
        or schedule.reset_optimizer
    ):

        optimizer = create_optimizer(
            model=model,
            learning_rate=(
                schedule.learning_rate
            ),
            weight_decay=(
                schedule.weight_decay
            ),
        )

    else:

        set_optimizer_learning_rate(
            optimizer=optimizer,
            learning_rate=(
                schedule.learning_rate
            ),
        )

    main_criterion = (
        nn.CrossEntropyLoss()
    )

    aux_criterion = (
        nn.CrossEntropyLoss()
    )

    loader = _double_head_loader(
        X=X,
        y_main=y_main,
        y_aux=y_aux,
        batch_size=(
            schedule.batch_size
        ),
        shuffle=True,
    )

    mean_epoch_losses: list[
        float
    ] = []

    validation_losses: list[
        float
    ] = []

    best_validation_loss = float(
        "inf"
    )

    best_epoch = 0

    best_model_state = None
    best_optimizer_state = None

    epochs_without_improvement = 0

    stopped_early = False

    prefix = (
        f"[{log_prefix}] "
        if log_prefix
        else ""
    )

    model.train()

    for epoch_index in range(
        schedule.epochs
    ):

        epoch = (
            epoch_index + 1
        )

        total_loss = 0.0
        total_samples = 0

        for (
            batch_X,
            batch_main,
            batch_aux,
        ) in loader:

            batch_X = batch_X.to(
                device
            )

            batch_main = batch_main.to(
                device
            )

            batch_aux = batch_aux.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            (
                main_logits,
                aux_logits,
            ) = model.forward_both(
                batch_X
            )

            main_loss = (
                main_criterion(
                    main_logits,
                    batch_main,
                )
            )

            aux_loss = (
                aux_criterion(
                    aux_logits,
                    batch_aux,
                )
            )

            loss = (
                main_loss
                + aux_loss_weight
                * aux_loss
            )

            loss.backward()

            optimizer.step()

            batch_n = int(
                len(
                    batch_X
                )
            )

            total_loss += (
                float(
                    loss.detach().item()
                )
                * batch_n
            )

            total_samples += (
                batch_n
            )

        train_loss = (
            total_loss
            / max(
                total_samples,
                1,
            )
        )

        mean_epoch_losses.append(
            train_loss
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        if has_validation:

            val_loss = (
                _double_head_validation_loss(
                    model=model,
                    X=X_validation,
                    y_main=(
                        y_main_validation
                    ),
                    y_aux=(
                        y_aux_validation
                    ),
                    batch_size=(
                        schedule.batch_size
                    ),
                    aux_loss_weight=(
                        aux_loss_weight
                    ),
                    device=device,
                )
            )

            validation_losses.append(
                val_loss
            )

            if schedule.early_stopping_enabled:

                improved = (
                    val_loss
                    < (
                        best_validation_loss
                        - schedule.early_stopping_min_delta
                    )
                )

            else:

                improved = (
                    val_loss
                    < best_validation_loss
                )

            if improved:

                best_validation_loss = (
                    val_loss
                )

                best_epoch = epoch

                best_model_state = (
                    copy.deepcopy(
                        model.state_dict()
                    )
                )

                best_optimizer_state = (
                    copy.deepcopy(
                        optimizer.state_dict()
                    )
                )

                epochs_without_improvement = 0

            elif schedule.early_stopping_enabled:

                epochs_without_improvement += 1

            if schedule.early_stopping_enabled:

                print(
                    f"{prefix}"
                    f"epoch {epoch}/{schedule.epochs} | "
                    f"train_loss={train_loss:.6f} | "
                    f"val_loss={val_loss:.6f} | "
                    f"best_val_loss="
                    f"{best_validation_loss:.6f} | "
                    f"best_epoch={best_epoch} | "
                    f"patience="
                    f"{epochs_without_improvement}/"
                    f"{schedule.early_stopping_patience}"
                )

            else:

                print(
                    f"{prefix}"
                    f"epoch {epoch}/{schedule.epochs} | "
                    f"train_loss={train_loss:.6f} | "
                    f"val_loss={val_loss:.6f}"
                )

            # =================================================
            # EARLY STOP
            # =================================================

            if (
                schedule.early_stopping_enabled
                and epochs_without_improvement
                >= schedule.early_stopping_patience
            ):

                stopped_early = True

                print(
                    f"{prefix}"
                    f"EARLY STOP at epoch {epoch} | "
                    f"best_epoch={best_epoch} | "
                    f"best_val_loss="
                    f"{best_validation_loss:.6f}"
                )

                break

        else:

            print(
                f"{prefix}"
                f"epoch {epoch}/{schedule.epochs} | "
                f"train_loss={train_loss:.6f}"
            )

    executed_epochs = len(
        mean_epoch_losses
    )

    # ========================================================
    # RESTORE BEST CHECKPOINT
    # ========================================================

    if (
        schedule.early_stopping_enabled
        and best_model_state is not None
    ):

        model.load_state_dict(
            best_model_state
        )

        optimizer.load_state_dict(
            best_optimizer_state
        )

    # ========================================================
    # FINAL LOG
    # ========================================================

    if has_validation:

        print(
            f"{prefix}"
            f"TRAINING FINISHED | "
            f"executed_epochs={executed_epochs} | "
            f"best_epoch={best_epoch} | "
            f"best_val_loss="
            f"{best_validation_loss:.6f} | "
            f"stopped_early={stopped_early}"
        )

    else:

        print(
            f"{prefix}"
            f"TRAINING FINISHED | "
            f"executed_epochs={executed_epochs}"
        )

    return TrainingResult(
        model=model,

        optimizer=optimizer,

        epochs=executed_epochs,

        learning_rate=float(
            schedule.learning_rate
        ),

        mean_epoch_losses=(
            mean_epoch_losses
        ),

        validation_losses=(
            validation_losses
        ),

        best_epoch=best_epoch,

        best_validation_loss=(
            best_validation_loss
        ),

        stopped_early=(
            stopped_early
        ),
    )

# ============================================================
# CONFIG CONVENIENCE
# ============================================================

def get_aux_loss_weight(
    config: ExperimentConfig,
) -> float:
    """
    Read the Double Head auxiliary-loss weight.
    """

    return float(
        config.get(
            "training",
            "double_head",
            "aux_loss_weight",
        )
    )