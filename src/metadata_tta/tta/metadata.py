from __future__ import annotations

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


class MetadataTTA(
    TTAMethod
):
    """
    Raw metadata-supervised test-time adaptation.

    Only the residual online adapter is trainable.

    For each observation/batch:

        1. compute auxiliary metadata CE;
        2. normalize it by log(n_aux_classes);
        3. skip adaptation when normalized loss is below
           the configured threshold;
        4. otherwise optimize:

               L_aux
             + lambda_anchor * ||theta - theta_source||^2

        5. clip adapter gradients;
        6. update adapter parameters.

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

        if self.normalized_aux_loss_min < 0.0:
            raise ValueError(
                "metadata.normalized_aux_loss_min must be >= 0."
            )

        if self.normalized_aux_loss_max <= self.normalized_aux_loss_min:
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

    def _log_trainable_scope(
        self,
    ) -> None:
        """
        Validate the V3 Metadata-TTA trainable scope.

        Metadata adaptation is allowed to modify only the residual
        online adapter.
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

        if trainable_parameter_ids != adapter_parameter_ids:
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
        Configure the V3 Metadata-TTA adaptation scope.

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

        # Freeze the complete model first.
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        # Enable only the residual adapter.
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

        # Explicit scientific invariant.
        for name, parameter in self.model.named_parameters():

            should_be_trainable = name.startswith(
                "online_adapter."
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

        # Immutable source anchor for exactly the parameters that
        # Metadata TTA may adapt.
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

        Metadata TTA V3 adapts only the residual online adapter.
        The optimizer state is also cleared so adaptation restarts
        with fresh Adam statistics.

        This primitive is used by episodic, annual-reset, and
        drift-reset metadata variants.
        """

        if len(
            self.trainable_parameters
        ) != len(
            self.source_parameter_values
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

        # Fresh Adam state for every episodic sample.
        self.optimizer.zero_grad(
            set_to_none=True
        )

        self.optimizer.state.clear()

        # Keep Dropout disabled and BN running statistics fixed.
        self.model.eval()


    def reset_online_adapter_to_source(
        self,
    ) -> None:
        """
        Backward-compatible alias required by the current
        episodic Metadata evaluator.
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

        self.n_aux_window_pass = 0
        self.n_aux_below_window = 0
        self.n_aux_above_window = 0

        self.sum_aux_loss = 0.0
        self.sum_normalized_aux_loss = 0.0

        self.sum_gradient_norm = 0.0
        self.sum_parameter_delta = 0.0

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
                    "reason": "below_aux_loss_window",
                    "aux_loss": aux_loss_value,
                    "normalized_aux_loss": normalized_aux_loss,
                    "batch_size": int(len(x_tensor)),
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
                    "reason": "above_aux_loss_window",
                    "aux_loss": aux_loss_value,
                    "normalized_aux_loss": normalized_aux_loss,
                    "batch_size": int(len(x_tensor)),
                },
            )

        self.n_aux_window_pass += 1

        # ====================================================
        # ADAPT
        # ====================================================

        mean_gradient_norm = 0.0
        mean_parameter_delta = 0.0

        final_aux_loss = (
            aux_loss_value
        )

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

            anchor_loss = (
                self._source_anchor_loss()
            )

            total_loss = (
                aux_loss
                + self.regularization
                * anchor_loss
            )

            total_loss.backward()

            gradient_norm = (
                self._gradient_norm()
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

        return AdaptationResult(
            applied=True,
            diagnostics={
                "reason":
                    "updated",

                "aux_loss":
                    final_aux_loss,

                "normalized_aux_loss":
                    normalized_aux_loss,

                "gradient_norm":
                    mean_gradient_norm,

                "parameter_delta":
                    mean_parameter_delta,

                "batch_size":
                    int(
                        len(
                            x_tensor
                        )
                    ),
            },
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

        diagnostics.update(
            {
                "aux_window_pass_count": int(
                    self.n_aux_window_pass
                ),

                "aux_window_pass_rate": float(
                    self.n_aux_window_pass
                    / n_observations
                ),

                "aux_below_window_count": int(
                    self.n_aux_below_window
                ),

                "aux_below_window_rate": float(
                    self.n_aux_below_window
                    / n_observations
                ),

                "aux_above_window_count": int(
                    self.n_aux_above_window
                ),

                "aux_above_window_rate": float(
                    self.n_aux_above_window
                    / n_observations
                ),

                "normalized_aux_loss_min": float(
                    self.normalized_aux_loss_min
                ),

                "normalized_aux_loss_max": float(
                    self.normalized_aux_loss_max
                ),

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

                "mean_parameter_delta":
                    float(
                        self.sum_parameter_delta
                        / n_updates
                    ),
                "trainable_scope": "online_adapter",
                "n_trainable_parameters":
                    int(
                        sum(
                            parameter.numel()
                            for parameter
                            in self.trainable_parameters
                        )
                    ),
            }
        )

        return diagnostics