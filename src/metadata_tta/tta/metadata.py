from __future__ import annotations

import copy
import math
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import (
    AdaptationResult,
    EvaluationOrder,
    TTAMethod,
)

def compute_source_aux_gradient_norms(
    *,
    model: nn.Module,
    X: np.ndarray,
    y_aux: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Compute single-sample auxiliary gradient norms on frozen
    source-validation data.

    The supplied source model is NEVER modified:
    calibration is performed on a deepcopy.

    Norms are computed with respect to the online adapter,
    before gradient clipping, exactly matching the quantity
    used by MetadataTTA's gradient-norm gate.

    No main-task labels are used.
    No model update is performed.
    """

    if len(X) != len(y_aux):
        raise ValueError(
            "Gradient-norm calibration received different "
            "numbers of features and auxiliary labels."
        )

    if len(X) == 0:
        raise ValueError(
            "Gradient-norm calibration requires non-empty data."
        )

    calibration_model = copy.deepcopy(
        model
    ).to(device)

    calibration_model.eval()

    if not hasattr(
        calibration_model,
        "online_adapter",
    ):
        raise TypeError(
            "Gradient-norm calibration requires "
            "an online_adapter."
        )

    if not hasattr(
        calibration_model,
        "extract_shared_features",
    ):
        raise TypeError(
            "Gradient-norm calibration requires "
            "extract_shared_features()."
        )

    if not hasattr(
        calibration_model,
        "aux_head",
    ):
        raise TypeError(
            "Gradient-norm calibration requires an aux_head."
        )

    aux_head = calibration_model.aux_head

    if not hasattr(
        aux_head,
        "out_features",
    ):
        raise TypeError(
            "Gradient-norm calibration expects aux_head "
            "to expose out_features."
        )

    n_classes_aux = int(
        aux_head.out_features
    )

    for parameter in calibration_model.parameters():
        parameter.requires_grad_(False)

    adapter_parameters = list(
        calibration_model.online_adapter.parameters()
    )

    if not adapter_parameters:
        raise RuntimeError(
            "Gradient-norm calibration found no "
            "online_adapter parameters."
        )

    for parameter in adapter_parameters:
        parameter.requires_grad_(True)

    norms: list[float] = []

    for index in range(
        len(X)
    ):

        label = int(
            y_aux[index]
        )

        if (
            label < 0
            or label >= n_classes_aux
        ):
            continue

        x_tensor = torch.as_tensor(
            X[index:index + 1],
            dtype=torch.float32,
            device=device,
        )

        y_tensor = torch.as_tensor(
            [label],
            dtype=torch.long,
            device=device,
        )

        shared_features = (
            calibration_model
            .extract_shared_features(
                x_tensor
            )
        )

        aux_logits = (
            calibration_model
            .aux_head(
                shared_features
            )
        )

        aux_loss = F.cross_entropy(
            aux_logits,
            y_tensor,
        )

        gradients = torch.autograd.grad(
            outputs=aux_loss,
            inputs=adapter_parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        squared_norm = 0.0

        for gradient in gradients:

            if gradient is None:
                continue

            squared_norm += float(
                torch.sum(
                    gradient
                    .detach()
                    .float()
                    ** 2
                ).item()
            )

        norms.append(
            float(
                math.sqrt(
                    max(
                        squared_norm,
                        0.0,
                    )
                )
            )
        )

    if not norms:
        raise RuntimeError(
            "Gradient-norm calibration produced no valid norms."
        )

    result = np.asarray(
        norms,
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(
            result
        )
    ):
        raise RuntimeError(
            "Gradient-norm calibration produced "
            "non-finite values."
        )

    return result

class MetadataTTA(
    TTAMethod
):
    """
    Metadata-supervised test-time adaptation.

    Only the residual online adapter is trainable.

    For each observation/batch:

        1. compute auxiliary metadata CE;
        2. normalize it by log(n_aux_classes);
        3. apply the configured auxiliary-loss reliability window;
        4. if main preservation is enabled:
               a. obtain an immutable source-model main prediction;
               b. compute source confidence and top-1/top-2 classes;
               c. compute the auxiliary gradient;
               d. compute a main-margin guard gradient;
               e. if the two gradients conflict, remove the
                  conflicting auxiliary component proportionally
                  to source main confidence;
        5. optionally add source-anchor regularization;
        6. clip adapter gradients;
        7. update adapter parameters.

    No OOD main-task labels are used.
    """

    method_name = "metadata"

    requires_aux_labels = True

    evaluation_order = EvaluationOrder.ADAPT_THEN_PREDICT

    def __init__(
        self,
        source_model: nn.Module,
        config: Mapping[str, Any],
        device: torch.device,
    ) -> None:

        super().__init__(
            source_model=source_model,
            config=config,
            device=device,
        )

        # ====================================================
        # CONFIG
        # ====================================================

        self.learning_rate = float(
            self.config[
                "learning_rate"
            ]
        )

        self.batch_size = int(
            self.config.get(
                "batch_size",
                1,
            )
        )

        self.regularization = float(
            self.config[
                "regularization"
            ]
        )

        self.gradient_clip = float(
            self.config[
                "gradient_clip"
            ]
        )

        self.normalized_aux_loss_min = float(
            self.config.get(
                "normalized_aux_loss_min",
                0.0,
            )
        )

        self.normalized_aux_loss_max = float(
            self.config.get(
                "normalized_aux_loss_max",
                1.0,
            )
        )

        self.main_preservation_enabled = bool(
            self.config.get(
                "main_preservation_enabled",
                False,
            )
        )

        gradient_norm_max_config = (
            self.config.get(
                "gradient_norm_max",
                None,
            )
        )

        self.gradient_norm_max = (
            float("inf")
            if gradient_norm_max_config is None
            else float(
                gradient_norm_max_config
            )
        )

        gradient_norm_quantile_config = (
            self.config.get(
                "gradient_norm_quantile",
                None,
            )
        )

        self.gradient_norm_quantile = (
            None
            if gradient_norm_quantile_config is None
            else float(
                gradient_norm_quantile_config
            )
        )

        self.steps = int(
            self.config.get(
                "steps",
                1,
            )
        )

        self.epsilon = float(
            self.config.get(
                "epsilon",
                1.0e-12,
            )
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        if (
            self.gradient_norm_quantile is not None
            and not (
                0.0
                < self.gradient_norm_quantile
                < 1.0
            )
        ):
            raise ValueError(
                "metadata.gradient_norm_quantile must satisfy "
                "0 < q < 1 when enabled."
            )

        if (
            self.gradient_norm_quantile is not None
            and not math.isfinite(
                self.gradient_norm_max
            )
        ):
            raise ValueError(
                "metadata.gradient_norm_quantile requires "
                "a calibrated finite gradient_norm_max."
            )

        if self.learning_rate <= 0.0:
            raise ValueError(
                "metadata.learning_rate must be > 0."
            )

        if self.batch_size < 1:
            raise ValueError(
                "metadata.batch_size must be >= 1."
            )

        if self.regularization < 0.0:
            raise ValueError(
                "metadata.regularization must be >= 0."
            )

        if self.gradient_clip < 0.0:
            raise ValueError(
                "metadata.gradient_clip must be >= 0."
            )

        if self.steps < 1:
            raise ValueError(
                "metadata.steps must be >= 1."
            )

        if (
            math.isfinite(
                self.gradient_norm_max
            )
            and self.gradient_norm_max <= 0.0
        ):
            raise ValueError(
                "metadata.gradient_norm_max must be > 0 "
                "when enabled."
            )

        if (
            math.isfinite(
                self.gradient_norm_max
            )
            and self.steps != 1
        ):
            raise ValueError(
                "metadata.gradient_norm_max currently "
                "requires steps == 1 so a rejected update "
                "cannot occur after a partial multi-step "
                "adaptation."
            )

        if self.normalized_aux_loss_min < 0.0:
            raise ValueError(
                "metadata.normalized_aux_loss_min must be >= 0."
            )

        if (
            self.normalized_aux_loss_max
            <= self.normalized_aux_loss_min
        ):
            raise ValueError(
                "metadata.normalized_aux_loss_max must be "
                "> normalized_aux_loss_min."
            )

        # ====================================================
        # MODEL REQUIREMENTS
        # ====================================================

        if not hasattr(
            self.model,
            "online_adapter",
        ):
            raise TypeError(
                "MetadataTTA requires a model with "
                "an online_adapter."
            )

        if not hasattr(
            self.model,
            "extract_shared_features",
        ):
            raise TypeError(
                "MetadataTTA requires "
                "extract_shared_features()."
            )

        if not hasattr(
            self.model,
            "aux_head",
        ):
            raise TypeError(
                "MetadataTTA requires an aux_head."
            )

        # ====================================================
        # ONLY ONLINE ADAPTER TRAINABLE
        # ====================================================

        self._configure_trainable_parameters()

        # ====================================================
        # AUXILIARY TASK
        # ====================================================

        aux_head = self.model.aux_head

        if not hasattr(
            aux_head,
            "out_features",
        ):
            raise TypeError(
                "MetadataTTA expects aux_head to expose "
                "out_features."
            )

        self.n_classes_aux = int(
            aux_head.out_features
        )

        self.aux_loss_normalizer = float(
            math.log(
                max(
                    self.n_classes_aux,
                    2,
                )
            )
        )

        # ====================================================
        # OPTIMIZER
        # ====================================================

        self.optimizer = (
            self._build_optimizer()
        )

        # ====================================================
        # METHOD STATS
        # ====================================================

        self._reset_method_stats()

        self._log_trainable_scope()

    # ========================================================
    # TRAINABLE SCOPE
    # ========================================================

    def _log_trainable_scope(
        self,
    ) -> None:
        """
        Validate the Metadata-TTA trainable scope.

        Metadata adaptation is allowed to modify only the
        residual online adapter.
        """

        adapter_parameter_ids = {
            id(parameter)
            for parameter
            in self.model.online_adapter.parameters()
        }

        trainable_parameter_ids = {
            id(parameter)
            for parameter
            in self.trainable_parameters
        }

        if (
            trainable_parameter_ids
            != adapter_parameter_ids
        ):
            raise RuntimeError(
                "Metadata TTA trainable-scope violation: "
                "only online_adapter parameters may be trainable."
            )

        unexpected_trainable = [
            name
            for name, parameter
            in self.model.named_parameters()
            if (
                parameter.requires_grad
                and id(parameter)
                not in adapter_parameter_ids
            )
        ]

        if unexpected_trainable:
            raise RuntimeError(
                "Metadata TTA found unexpected trainable "
                "parameters outside online_adapter: "
                f"{unexpected_trainable}"
            )

    # ========================================================
    # MODEL CONFIGURATION
    # ========================================================

    def _configure_trainable_parameters(
        self,
    ) -> None:
        """
        Configure the Metadata-TTA adaptation scope.

        Trainable:
            online_adapter

        Frozen:
            feature_block
            main_head
            aux_head

        The complete model remains in eval mode so Dropout is
        disabled and BatchNorm running statistics stay frozen.
        """

        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        for parameter in (
            self.model.online_adapter.parameters()
        ):
            parameter.requires_grad_(True)

        self.trainable_parameters = list(
            self.model.online_adapter.parameters()
        )

        if not self.trainable_parameters:
            raise RuntimeError(
                "MetadataTTA found no online_adapter parameters."
            )

        for name, parameter in self.model.named_parameters():

            should_be_trainable = (
                name.startswith(
                    "online_adapter."
                )
            )

            if (
                parameter.requires_grad
                != should_be_trainable
            ):
                raise RuntimeError(
                    "Metadata TTA trainable-parameter invariant "
                    f"failed for {name!r}: "
                    f"requires_grad={parameter.requires_grad}, "
                    f"expected={should_be_trainable}."
                )

        # Immutable source values for exactly the parameters
        # Metadata TTA is allowed to modify.
        self.source_parameter_values = [
            parameter.detach().clone()
            for parameter
            in self.trainable_parameters
        ]

    def _build_optimizer(
        self,
    ) -> torch.optim.Optimizer:

        return torch.optim.Adam(
            self.trainable_parameters,
            lr=self.learning_rate,
        )

    def reset_adapted_parameters_to_source(
        self,
    ) -> None:
        """
        Restore the online adapter to its source state.

        The optimizer state is cleared so adaptation restarts
        with fresh Adam statistics.
        """

        if (
            len(
                self.trainable_parameters
            )
            != len(
                self.source_parameter_values
            )
        ):
            raise RuntimeError(
                "Mismatch between trainable parameters "
                "and stored source parameter values."
            )

        with torch.no_grad():

            for (
                parameter,
                source_value,
            ) in zip(
                self.trainable_parameters,
                self.source_parameter_values,
            ):

                parameter.copy_(
                    source_value
                )

                parameter.grad = None

        self.optimizer.zero_grad(
            set_to_none=True
        )

        self.optimizer.state.clear()

        self.model.eval()

    def reset_online_adapter_to_source(
        self,
    ) -> None:
        """
        Backward-compatible alias required by the episodic
        Metadata evaluator.
        """

        self.reset_adapted_parameters_to_source()

    # ========================================================
    # RESET
    # ========================================================

    def _reset_adaptation_state(
        self,
    ) -> None:

        self._configure_trainable_parameters()

        self.optimizer = (
            self._build_optimizer()
        )

        self._reset_method_stats()

    # ========================================================
    # METHOD STATS
    # ========================================================

    def _reset_method_stats(
        self,
    ) -> None:

        # -------------------------------
        # Reliability window
        # -------------------------------

        self.n_aux_window_pass = 0
        self.n_aux_below_window = 0
        self.n_aux_above_window = 0

        self.sum_aux_loss = 0.0
        self.sum_normalized_aux_loss = 0.0

        self.sum_gradient_norm = 0.0
        self.sum_parameter_delta = 0.0

        # -------------------------------
        # Gradient-norm reliability gate
        # -------------------------------

        self.n_gradient_norm_rejected = 0
        self.sum_rejected_gradient_norm = 0.0

        # -------------------------------
        # Main-preservation diagnostics
        # -------------------------------

        self.n_main_reference_evaluations = 0
        self.n_main_guard_evaluations = 0
        self.n_main_gradient_conflicts = 0

        self.sum_source_main_confidence = 0.0
        self.sum_source_main_margin = 0.0

        self.sum_aux_guard_cosine = 0.0

        self.sum_projection_strength = 0.0

        self.sum_raw_aux_gradient_norm = 0.0
        self.sum_safe_aux_gradient_norm = 0.0
        self.sum_guard_gradient_norm = 0.0

        self.sum_removed_gradient_fraction = 0.0

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict_logits(
        self,
        x: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:

        x_tensor = self._as_feature_tensor(
            x
        )

        self.model.eval()

        with torch.no_grad():

            logits = self.model(
                x_tensor
            )

        return (
            logits.detach()
        )

    # ========================================================
    # AUXILIARY LOSS
    # ========================================================

    def _auxiliary_loss(
        self,
        x_tensor: torch.Tensor,
        y_aux_tensor: torch.Tensor,
    ) -> torch.Tensor:

        shared_features = (
            self.model
            .extract_shared_features(
                x_tensor
            )
        )

        aux_logits = (
            self.model
            .aux_head(
                shared_features
            )
        )

        return F.cross_entropy(
            aux_logits,
            y_aux_tensor,
        )

    # ========================================================
    # SOURCE ANCHOR
    # ========================================================

    def _source_anchor_loss(
        self,
    ) -> torch.Tensor:
        """
        Squared L2 distance between current adapter parameters
        and their frozen source values.
        """

        anchor_loss = torch.zeros(
            (),
            dtype=torch.float32,
            device=self.device,
        )

        for (
            parameter,
            source_value,
        ) in zip(
            self.trainable_parameters,
            self.source_parameter_values,
        ):

            anchor_loss = (
                anchor_loss
                + torch.sum(
                    (
                        parameter
                        - source_value
                    )
                    ** 2
                )
            )

        return anchor_loss

    # ========================================================
    # GENERIC GRADIENT HELPERS
    # ========================================================

    @staticmethod
    def _gradient_list_norm(
        gradients: list[torch.Tensor],
    ) -> float:

        squared_norm = 0.0

        for gradient in gradients:

            squared_norm += float(
                torch.sum(
                    gradient.detach().float() ** 2
                ).item()
            )

        return float(
            math.sqrt(
                max(
                    squared_norm,
                    0.0,
                )
            )
        )

    @staticmethod
    def _gradient_list_dot(
        first: list[torch.Tensor],
        second: list[torch.Tensor],
    ) -> float:

        if len(first) != len(second):
            raise RuntimeError(
                "Gradient-list size mismatch."
            )

        dot_product = 0.0

        for (
            first_gradient,
            second_gradient,
        ) in zip(
            first,
            second,
        ):

            dot_product += float(
                torch.sum(
                    first_gradient.detach().float()
                    * second_gradient.detach().float()
                ).item()
            )

        return float(
            dot_product
        )

    def _autograd_gradient_list(
        self,
        loss: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Compute gradients with respect to the trainable adapter.

        Missing gradients are represented explicitly as zeros so
        all gradient lists have identical structure.
        """

        gradients = torch.autograd.grad(
            outputs=loss,
            inputs=self.trainable_parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        result: list[torch.Tensor] = []

        for (
            parameter,
            gradient,
        ) in zip(
            self.trainable_parameters,
            gradients,
        ):

            if gradient is None:
                result.append(
                    torch.zeros_like(
                        parameter
                    )
                )
            else:
                result.append(
                    gradient.detach()
                )

        return result

    # ========================================================
    # IMMUTABLE SOURCE MAIN REFERENCE
    # ========================================================

    def _source_main_reference(
        self,
        x_tensor: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Obtain the main-task prediction produced by the immutable
        source adapter.

        Only online_adapter can change during Metadata TTA.
        Therefore temporarily restoring its source parameters is
        sufficient to reconstruct the original source-model
        prediction without allocating a second complete model.

        The current online adapter is restored immediately after
        the source forward pass. Optimizer state is untouched.
        """

        current_parameter_values = [
            parameter.detach().clone()
            for parameter
            in self.trainable_parameters
        ]

        try:

            with torch.no_grad():

                for (
                    parameter,
                    source_value,
                ) in zip(
                    self.trainable_parameters,
                    self.source_parameter_values,
                ):

                    parameter.copy_(
                        source_value
                    )

                self.model.eval()

                source_logits = self.model(
                    x_tensor
                )

                if (
                    source_logits.ndim != 2
                    or source_logits.shape[1] < 2
                ):
                    raise RuntimeError(
                        "Main-preserving TTA requires at least "
                        "two main-task output classes."
                    )

                probabilities = F.softmax(
                    source_logits,
                    dim=1,
                )

                log_probabilities = torch.log(
                    probabilities.clamp_min(
                        self.epsilon
                    )
                )

                entropy = -torch.sum(
                    probabilities
                    * log_probabilities,
                    dim=1,
                )

                n_main_classes = int(
                    source_logits.shape[1]
                )

                entropy_normalizer = float(
                    math.log(
                        max(
                            n_main_classes,
                            2,
                        )
                    )
                )

                source_confidence = (
                    1.0
                    - entropy
                    / max(
                        entropy_normalizer,
                        self.epsilon,
                    )
                )

                source_confidence = (
                    source_confidence.clamp(
                        min=0.0,
                        max=1.0,
                    )
                )

                top_values, top_indices = (
                    torch.topk(
                        source_logits,
                        k=2,
                        dim=1,
                    )
                )

                source_margin = (
                    top_values[:, 0]
                    - top_values[:, 1]
                )

                result = {
                    "top1_indices":
                        top_indices[:, 0]
                        .detach()
                        .clone(),

                    "top2_indices":
                        top_indices[:, 1]
                        .detach()
                        .clone(),

                    "confidence":
                        float(
                            source_confidence
                            .mean()
                            .item()
                        ),

                    "margin":
                        float(
                            source_margin
                            .mean()
                            .item()
                        ),
                }

        finally:

            with torch.no_grad():

                for (
                    parameter,
                    current_value,
                ) in zip(
                    self.trainable_parameters,
                    current_parameter_values,
                ):

                    parameter.copy_(
                        current_value
                    )

            self.model.eval()

        return result

    # ========================================================
    # MAIN-MARGIN GUARD
    # ========================================================

    def _main_guard_gradient(
        self,
        x_tensor: torch.Tensor,
        top1_indices: torch.Tensor,
        top2_indices: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Gradient of the negative source-class main margin.

        Let

            M = z_source_top1 - z_source_top2

        and

            L_guard = -M.

        Gradient descent on L_guard increases the source-model
        top1/top2 margin.

        The resulting gradient therefore represents a locally
        main-compatible descent direction.
        """

        self.model.eval()

        main_logits = self.model(
            x_tensor
        )

        if (
            main_logits.ndim != 2
            or main_logits.shape[1] < 2
        ):
            raise RuntimeError(
                "Main-preserving TTA requires at least "
                "two main-task output classes."
            )

        top1_logits = torch.gather(
            main_logits,
            dim=1,
            index=(
                top1_indices
                .reshape(-1, 1)
            ),
        ).reshape(-1)

        top2_logits = torch.gather(
            main_logits,
            dim=1,
            index=(
                top2_indices
                .reshape(-1, 1)
            ),
        ).reshape(-1)

        main_margin = (
            top1_logits
            - top2_logits
        )

        guard_loss = -torch.mean(
            main_margin
        )

        return self._autograd_gradient_list(
            guard_loss
        )

    # ========================================================
    # MAIN-PRESERVING GRADIENT SURGERY
    # ========================================================

    def _main_preserving_aux_gradient(
        self,
        raw_aux_gradients: list[torch.Tensor],
        guard_gradients: list[torch.Tensor],
        source_confidence: float,
    ) -> tuple[
        list[torch.Tensor],
        dict[str, Any],
    ]:
        """
        Confidence-weighted gradient surgery.

        If the auxiliary gradient and main guard gradient are
        compatible, keep the auxiliary gradient unchanged.

        If they conflict:

            g_aux · g_guard < 0

        remove a confidence-weighted fraction of the conflicting
        component:

            g_safe =
                g_aux
                - c *
                  (g_aux · g_guard)
                  / (||g_guard||^2 + eps)
                  * g_guard

        where c is the immutable source-model main confidence.
        """

        raw_aux_norm = (
            self._gradient_list_norm(
                raw_aux_gradients
            )
        )

        guard_norm = (
            self._gradient_list_norm(
                guard_gradients
            )
        )

        dot_product = (
            self._gradient_list_dot(
                raw_aux_gradients,
                guard_gradients,
            )
        )

        denominator = (
            raw_aux_norm
            * guard_norm
        )

        if denominator > self.epsilon:

            cosine = float(
                max(
                    -1.0,
                    min(
                        1.0,
                        dot_product
                        / denominator,
                    ),
                )
            )

        else:

            cosine = 0.0

        conflict = bool(
            cosine <= -0.02
            and guard_norm > self.epsilon
            and raw_aux_norm > self.epsilon
            and source_confidence >= 0.95
        )

        projection_strength = 0.0

        if conflict:

            projection_strength = float(
                min(
                    max(
                        source_confidence,
                        0.0,
                    ),
                    1.0,
                )
            )

            guard_squared_norm = (
                guard_norm ** 2
            )

            projection_coefficient = float(
                projection_strength
                * dot_product
                / (
                    guard_squared_norm
                    + self.epsilon
                )
            )

            safe_gradients = [
                (
                    raw_gradient
                    - projection_coefficient
                    * guard_gradient
                )
                for (
                    raw_gradient,
                    guard_gradient,
                ) in zip(
                    raw_aux_gradients,
                    guard_gradients,
                )
            ]

        else:

            safe_gradients = [
                gradient.clone()
                for gradient
                in raw_aux_gradients
            ]

        safe_aux_norm = (
            self._gradient_list_norm(
                safe_gradients
            )
        )

        removed_gradients = [
            (
                raw_gradient
                - safe_gradient
            )
            for (
                raw_gradient,
                safe_gradient,
            ) in zip(
                raw_aux_gradients,
                safe_gradients,
            )
        ]

        removed_norm = (
            self._gradient_list_norm(
                removed_gradients
            )
        )

        if raw_aux_norm > self.epsilon:

            removed_fraction = float(
                removed_norm
                / raw_aux_norm
            )

        else:

            removed_fraction = 0.0

        diagnostics = {
            "aux_guard_cosine":
                float(
                    cosine
                ),

            "main_gradient_conflict":
                bool(
                    conflict
                ),

            "projection_strength":
                float(
                    projection_strength
                ),

            "raw_aux_gradient_norm":
                float(
                    raw_aux_norm
                ),

            "safe_aux_gradient_norm":
                float(
                    safe_aux_norm
                ),

            "guard_gradient_norm":
                float(
                    guard_norm
                ),

            "removed_gradient_fraction":
                float(
                    removed_fraction
                ),
        }

        return (
            safe_gradients,
            diagnostics,
        )

    # ========================================================
    # ASSIGN UPDATE GRADIENT
    # ========================================================

    def _assign_update_gradients(
        self,
        auxiliary_gradients: list[torch.Tensor],
        anchor_gradients: list[torch.Tensor] | None,
    ) -> None:
        """
        Write the final update gradients into parameter.grad.

        Gradient surgery is applied only to the metadata
        auxiliary gradient.

        Source-anchor regularization, when enabled, is added
        afterwards unchanged.
        """

        if (
            len(
                auxiliary_gradients
            )
            != len(
                self.trainable_parameters
            )
        ):
            raise RuntimeError(
                "Auxiliary-gradient size mismatch."
            )

        if (
            anchor_gradients is not None
            and len(
                anchor_gradients
            )
            != len(
                self.trainable_parameters
            )
        ):
            raise RuntimeError(
                "Anchor-gradient size mismatch."
            )

        for index, parameter in enumerate(
            self.trainable_parameters
        ):

            final_gradient = (
                auxiliary_gradients[
                    index
                ]
                .detach()
                .clone()
            )

            if anchor_gradients is not None:

                final_gradient = (
                    final_gradient
                    + self.regularization
                    * anchor_gradients[
                        index
                    ].detach()
                )

            parameter.grad = (
                final_gradient
            )

    # ========================================================
    # NORMS
    # ========================================================

    def _gradient_norm(
        self,
    ) -> float:

        squared_norm = 0.0

        for parameter in (
            self.trainable_parameters
        ):

            if parameter.grad is None:
                continue

            squared_norm += float(
                torch.sum(
                    parameter.grad
                    .detach()
                    .float()
                    ** 2
                ).item()
            )

        return float(
            math.sqrt(
                max(
                    squared_norm,
                    0.0,
                )
            )
        )

    @staticmethod
    def _parameter_delta_norm(
        parameters: list[
            nn.Parameter
        ],
        before: list[
            torch.Tensor
        ],
    ) -> float:

        squared_norm = 0.0

        for (
            parameter,
            previous,
        ) in zip(
            parameters,
            before,
        ):

            delta = (
                parameter
                .detach()
                .float()
                - previous
            )

            squared_norm += float(
                torch.sum(
                    delta ** 2
                ).item()
            )

        return float(
            math.sqrt(
                max(
                    squared_norm,
                    0.0,
                )
            )
        )

    # ========================================================
    # OBSERVE / UPDATE
    # ========================================================

    def observe(
        self,
        x: np.ndarray | torch.Tensor,
        y_aux: (
            np.ndarray
            | torch.Tensor
            | int
            | None
        ) = None,
    ) -> AdaptationResult:

        self._record_observation()

        if y_aux is None:
            raise ValueError(
                "MetadataTTA requires auxiliary labels."
            )

        x_tensor = self._as_feature_tensor(
            x
        )

        y_aux_tensor = self._as_label_tensor(
            y_aux
        )

        if (
            len(
                x_tensor
            )
            != len(
                y_aux_tensor
            )
        ):
            raise ValueError(
                "MetadataTTA received different batch sizes "
                "for x and y_aux."
            )

        if torch.any(
            y_aux_tensor < 0
        ) or torch.any(
            y_aux_tensor
            >= self.n_classes_aux
        ):
            return AdaptationResult(
                applied=False,
                diagnostics={
                    "reason":
                        "invalid_aux_label",
                },
            )

        # ====================================================
        # AUX LOSS GATE
        # ====================================================

        self.model.eval()

        with torch.no_grad():

            gate_loss = (
                self._auxiliary_loss(
                    x_tensor=x_tensor,
                    y_aux_tensor=(
                        y_aux_tensor
                    ),
                )
            )

        aux_loss_value = float(
            gate_loss.item()
        )

        normalized_aux_loss = float(
            aux_loss_value
            / max(
                self.aux_loss_normalizer,
                self.epsilon,
            )
        )

        self.sum_aux_loss += (
            aux_loss_value
        )

        self.sum_normalized_aux_loss += (
            normalized_aux_loss
        )

        if (
            normalized_aux_loss
            < self.normalized_aux_loss_min
        ):
            self.n_aux_below_window += 1

            return AdaptationResult(
                applied=False,
                diagnostics={
                    "reason":
                        "below_aux_loss_window",

                    "aux_loss":
                        aux_loss_value,

                    "normalized_aux_loss":
                        normalized_aux_loss,

                    "batch_size":
                        int(
                            len(
                                x_tensor
                            )
                        ),
                },
            )

        if (
            normalized_aux_loss
            > self.normalized_aux_loss_max
        ):
            self.n_aux_above_window += 1

            return AdaptationResult(
                applied=False,
                diagnostics={
                    "reason":
                        "above_aux_loss_window",

                    "aux_loss":
                        aux_loss_value,

                    "normalized_aux_loss":
                        normalized_aux_loss,

                    "batch_size":
                        int(
                            len(
                                x_tensor
                            )
                        ),
                },
            )

        self.n_aux_window_pass += 1

        # ====================================================
        # IMMUTABLE SOURCE MAIN REFERENCE
        # ====================================================

        source_reference: (
            dict[str, Any]
            | None
        ) = None

        if self.main_preservation_enabled:

            source_reference = (
                self._source_main_reference(
                    x_tensor
                )
            )

            self.n_main_reference_evaluations += 1

            self.sum_source_main_confidence += (
                float(
                    source_reference[
                        "confidence"
                    ]
                )
            )

            self.sum_source_main_margin += (
                float(
                    source_reference[
                        "margin"
                    ]
                )
            )

        # ====================================================
        # ADAPT
        # ====================================================

        mean_gradient_norm = 0.0
        mean_parameter_delta = 0.0

        final_aux_loss = (
            aux_loss_value
        )

        local_guard_evaluations = 0
        local_conflicts = 0

        local_sum_cosine = 0.0
        local_sum_projection_strength = 0.0
        local_sum_removed_fraction = 0.0

        for _ in range(
            self.steps
        ):

            parameters_before = [
                parameter
                .detach()
                .clone()
                .float()
                for parameter
                in self.trainable_parameters
            ]

            self.optimizer.zero_grad(
                set_to_none=True
            )

            aux_loss = (
                self._auxiliary_loss(
                    x_tensor=x_tensor,
                    y_aux_tensor=(
                        y_aux_tensor
                    ),
                )
            )

            # ================================================
            # V4 MAIN-PRESERVING PATH
            # ================================================

            if self.main_preservation_enabled:

                if source_reference is None:
                    raise RuntimeError(
                        "Main-preserving TTA is enabled "
                        "without a source main reference."
                    )

                raw_aux_gradients = (
                    self._autograd_gradient_list(
                        aux_loss
                    )
                )

                guard_gradients = (
                    self._main_guard_gradient(
                        x_tensor=x_tensor,
                        top1_indices=(
                            source_reference[
                                "top1_indices"
                            ]
                        ),
                        top2_indices=(
                            source_reference[
                                "top2_indices"
                            ]
                        ),
                    )
                )

                (
                    safe_aux_gradients,
                    surgery_diagnostics,
                ) = (
                    self
                    ._main_preserving_aux_gradient(
                        raw_aux_gradients=(
                            raw_aux_gradients
                        ),
                        guard_gradients=(
                            guard_gradients
                        ),
                        source_confidence=float(
                            source_reference[
                                "confidence"
                            ]
                        ),
                    )
                )

                anchor_gradients: (
                    list[torch.Tensor]
                    | None
                ) = None

                if self.regularization > 0.0:

                    anchor_loss = (
                        self._source_anchor_loss()
                    )

                    anchor_gradients = (
                        self._autograd_gradient_list(
                            anchor_loss
                        )
                    )

                self._assign_update_gradients(
                    auxiliary_gradients=(
                        safe_aux_gradients
                    ),
                    anchor_gradients=(
                        anchor_gradients
                    ),
                )

                # --------------------------------------------
                # Global surgery diagnostics
                # --------------------------------------------

                self.n_main_guard_evaluations += 1

                if bool(
                    surgery_diagnostics[
                        "main_gradient_conflict"
                    ]
                ):
                    self.n_main_gradient_conflicts += 1

                self.sum_aux_guard_cosine += (
                    float(
                        surgery_diagnostics[
                            "aux_guard_cosine"
                        ]
                    )
                )

                self.sum_projection_strength += (
                    float(
                        surgery_diagnostics[
                            "projection_strength"
                        ]
                    )
                )

                self.sum_raw_aux_gradient_norm += (
                    float(
                        surgery_diagnostics[
                            "raw_aux_gradient_norm"
                        ]
                    )
                )

                self.sum_safe_aux_gradient_norm += (
                    float(
                        surgery_diagnostics[
                            "safe_aux_gradient_norm"
                        ]
                    )
                )

                self.sum_guard_gradient_norm += (
                    float(
                        surgery_diagnostics[
                            "guard_gradient_norm"
                        ]
                    )
                )

                self.sum_removed_gradient_fraction += (
                    float(
                        surgery_diagnostics[
                            "removed_gradient_fraction"
                        ]
                    )
                )

                # --------------------------------------------
                # Per-observation surgery diagnostics
                # --------------------------------------------

                local_guard_evaluations += 1

                if bool(
                    surgery_diagnostics[
                        "main_gradient_conflict"
                    ]
                ):
                    local_conflicts += 1

                local_sum_cosine += float(
                    surgery_diagnostics[
                        "aux_guard_cosine"
                    ]
                )

                local_sum_projection_strength += float(
                    surgery_diagnostics[
                        "projection_strength"
                    ]
                )

                local_sum_removed_fraction += float(
                    surgery_diagnostics[
                        "removed_gradient_fraction"
                    ]
                )

            # ================================================
            # EXACT V3 PATH WHEN PRESERVATION IS DISABLED
            # ================================================

            else:

                anchor_loss = (
                    self._source_anchor_loss()
                )

                total_loss = (
                    aux_loss
                    + self.regularization
                    * anchor_loss
                )

                total_loss.backward()

            # ================================================
            # COMMON OPTIMIZER STEP
            # ================================================

            gradient_norm = (
                self._gradient_norm()
            )

            # ================================================
            # GRADIENT-NORM RELIABILITY GATE
            #
            # Reject the metadata update BEFORE clipping and
            # BEFORE optimizer.step().
            #
            # The threshold is applied to exactly the same
            # pre-clipping gradient norm used by the paired
            # episodic diagnostic.
            # ================================================

            if (
                gradient_norm
                > self.gradient_norm_max
            ):

                self.n_gradient_norm_rejected += 1

                self.sum_rejected_gradient_norm += (
                    gradient_norm
                )

                # No optimizer state or model parameter must
                # change for a rejected sample.
                self.optimizer.zero_grad(
                    set_to_none=True
                )

                return AdaptationResult(
                    applied=False,
                    diagnostics={
                        "reason":
                            "above_gradient_norm_max",

                        "aux_loss":
                            float(
                                aux_loss
                                .detach()
                                .item()
                            ),

                        "normalized_aux_loss":
                            normalized_aux_loss,

                        "gradient_norm":
                            float(
                                gradient_norm
                            ),

                        "gradient_norm_max":
                            float(
                                self.gradient_norm_max
                            ),

                        "gradient_norm_quantile":
                            (
                                None
                                if self.gradient_norm_quantile is None
                                else float(
                                    self.gradient_norm_quantile
                                )
                            ),

                        "parameter_delta":
                            0.0,

                        "batch_size":
                            int(
                                len(
                                    x_tensor
                                )
                            ),

                        "main_preservation_enabled":
                            bool(
                                self.main_preservation_enabled
                            ),
                    },
                )

            if self.gradient_clip > 0.0:

                torch.nn.utils.clip_grad_norm_(
                    self.trainable_parameters,
                    max_norm=(
                        self.gradient_clip
                    ),
                )

            self.optimizer.step()

            parameter_delta = (
                self._parameter_delta_norm(
                    parameters=(
                        self.trainable_parameters
                    ),
                    before=(
                        parameters_before
                    ),
                )
            )

            mean_gradient_norm += (
                gradient_norm
            )

            mean_parameter_delta += (
                parameter_delta
            )

            final_aux_loss = float(
                aux_loss
                .detach()
                .item()
            )

        mean_gradient_norm /= float(
            self.steps
        )

        mean_parameter_delta /= float(
            self.steps
        )

        self.sum_gradient_norm += (
            mean_gradient_norm
        )

        self.sum_parameter_delta += (
            mean_parameter_delta
        )

        self._record_update()

        result_diagnostics: dict[
            str,
            Any,
        ] = {
            "reason":
                "updated",

            "aux_loss":
                final_aux_loss,

            "normalized_aux_loss":
                normalized_aux_loss,

            "gradient_norm":
                mean_gradient_norm,

            "gradient_norm_quantile":
                    (
                        None
                        if self.gradient_norm_quantile is None
                        else float(
                            self.gradient_norm_quantile
                        )
                    ),

            "parameter_delta":
                mean_parameter_delta,

            "batch_size":
                int(
                    len(
                        x_tensor
                    )
                ),

            "main_preservation_enabled":
                bool(
                    self.main_preservation_enabled
                ),
        }

        if (
            self.main_preservation_enabled
            and source_reference is not None
        ):

            local_guard_denominator = max(
                local_guard_evaluations,
                1,
            )

            result_diagnostics.update(
                {
                    "source_main_confidence":
                        float(
                            source_reference[
                                "confidence"
                            ]
                        ),

                    "source_main_margin":
                        float(
                            source_reference[
                                "margin"
                            ]
                        ),

                    "mean_aux_guard_cosine":
                        float(
                            local_sum_cosine
                            / local_guard_denominator
                        ),

                    "main_gradient_conflict_count":
                        int(
                            local_conflicts
                        ),

                    "main_gradient_conflict_rate":
                        float(
                            local_conflicts
                            / local_guard_denominator
                        ),

                    "mean_projection_strength":
                        float(
                            local_sum_projection_strength
                            / local_guard_denominator
                        ),

                    "mean_removed_gradient_fraction":
                        float(
                            local_sum_removed_fraction
                            / local_guard_denominator
                        ),
                }
            )

        return AdaptationResult(
            applied=True,
            diagnostics=(
                result_diagnostics
            ),
        )

    # ========================================================
    # GLOBAL DIAGNOSTICS
    # ========================================================

    def diagnostics(
        self,
    ) -> dict[str, Any]:

        diagnostics = (
            super().diagnostics()
        )

        n_observations = max(
            self.number_of_observations,
            1,
        )

        n_updates = max(
            self.number_of_updates,
            1,
        )

        n_aux_window_pass = max(
            self.n_aux_window_pass,
            1,
        )

        n_gradient_norm_rejected = max(
            self.n_gradient_norm_rejected,
            1,
        )

        n_main_references = max(
            self.n_main_reference_evaluations,
            1,
        )

        n_main_guards = max(
            self.n_main_guard_evaluations,
            1,
        )

        diagnostics.update(
            {
                # --------------------------------------------
                # Reliability window
                # --------------------------------------------

                "aux_window_pass_count":
                    int(
                        self.n_aux_window_pass
                    ),

                "aux_window_pass_rate":
                    float(
                        self.n_aux_window_pass
                        / n_observations
                    ),

                "aux_below_window_count":
                    int(
                        self.n_aux_below_window
                    ),

                "aux_below_window_rate":
                    float(
                        self.n_aux_below_window
                        / n_observations
                    ),

                "aux_above_window_count":
                    int(
                        self.n_aux_above_window
                    ),

                "aux_above_window_rate":
                    float(
                        self.n_aux_above_window
                        / n_observations
                    ),

                "normalized_aux_loss_min":
                    float(
                        self.normalized_aux_loss_min
                    ),

                "normalized_aux_loss_max":
                    float(
                        self.normalized_aux_loss_max
                    ),

                # --------------------------------------------
                # Standard Metadata-TTA diagnostics
                # --------------------------------------------

                "mean_aux_loss":
                    float(
                        self.sum_aux_loss
                        / n_observations
                    ),

                "mean_normalized_aux_loss":
                    float(
                        self.sum_normalized_aux_loss
                        / n_observations
                    ),

                "mean_gradient_norm":
                    float(
                        self.sum_gradient_norm
                        / n_updates
                    ),

                # --------------------------------------------
                # Gradient-norm reliability gate
                # --------------------------------------------

                "gradient_norm_gate_enabled":
                    bool(
                        math.isfinite(
                            self.gradient_norm_max
                        )
                    ),

                "gradient_norm_max":
                    (
                        float(
                            self.gradient_norm_max
                        )
                        if math.isfinite(
                            self.gradient_norm_max
                        )
                        else None
                    ),

                "gradient_norm_rejected_count":
                    int(
                        self.n_gradient_norm_rejected
                    ),

                "gradient_norm_rejected_rate_among_aux_window_pass":
                    float(
                        self.n_gradient_norm_rejected
                        / n_aux_window_pass
                    ),

                "mean_rejected_gradient_norm":
                    float(
                        self.sum_rejected_gradient_norm
                        / n_gradient_norm_rejected
                    ),

                "trainable_scope":
                    "online_adapter",

                "n_trainable_parameters":
                    int(
                        sum(
                            parameter.numel()
                            for parameter
                            in self.trainable_parameters
                        )
                    ),

                # --------------------------------------------
                # V4 main-preservation diagnostics
                # --------------------------------------------

                "main_preservation_enabled":
                    bool(
                        self.main_preservation_enabled
                    ),

                "main_reference_evaluation_count":
                    int(
                        self.n_main_reference_evaluations
                    ),

                "main_guard_evaluation_count":
                    int(
                        self.n_main_guard_evaluations
                    ),

                "mean_source_main_confidence":
                    float(
                        self.sum_source_main_confidence
                        / n_main_references
                    ),

                "mean_source_main_margin":
                    float(
                        self.sum_source_main_margin
                        / n_main_references
                    ),

                "mean_aux_guard_cosine":
                    float(
                        self.sum_aux_guard_cosine
                        / n_main_guards
                    ),

                "main_gradient_conflict_count":
                    int(
                        self.n_main_gradient_conflicts
                    ),

                "main_gradient_conflict_rate":
                    float(
                        self.n_main_gradient_conflicts
                        / n_main_guards
                    ),

                "mean_projection_strength":
                    float(
                        self.sum_projection_strength
                        / n_main_guards
                    ),

                "mean_raw_aux_gradient_norm":
                    float(
                        self.sum_raw_aux_gradient_norm
                        / n_main_guards
                    ),

                "mean_safe_aux_gradient_norm":
                    float(
                        self.sum_safe_aux_gradient_norm
                        / n_main_guards
                    ),

                "mean_guard_gradient_norm":
                    float(
                        self.sum_guard_gradient_norm
                        / n_main_guards
                    ),

                "mean_removed_gradient_fraction":
                    float(
                        self.sum_removed_gradient_fraction
                        / n_main_guards
                    ),
            }
        )

        return diagnostics