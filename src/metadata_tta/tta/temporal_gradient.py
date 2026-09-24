from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import AdaptationResult
from .metadata import MetadataTTA


class TemporalGradientEMATTA(MetadataTTA):
    """
    Temporal Gradient Metadata TTA.

    Extends Raw Metadata TTA with a persistent gradient
    direction estimated causally through an exponential
    moving average of previous metadata gradients.

    During warmup:
        use the raw metadata gradient.

    After warmup:
        decompose the current raw gradient into a component
        parallel to the temporal EMA direction and an
        orthogonal component:

            g = g_parallel + g_orthogonal

        then use:

            g_temporal =
                g_parallel
                + orthogonal_scale * g_orthogonal

    The source-anchor gradient is added afterwards.

    Only online_adapter is trainable.

    No OOD main-task labels are used.
    """

    method_name = "temporal_gradient_ema"

    requires_aux_labels = True

    def __init__(
        self,
        source_model: nn.Module,
        config: Mapping[str, Any],
        device: torch.device,
    ) -> None:

        # MetadataTTA initializes:
        #
        # - model copy
        # - adapter-only trainability
        # - source anchor
        # - optimizer
        # - metadata gate
        #
        # The temporal-gradient config is deliberately
        # self-contained and therefore includes the common
        # Metadata-TTA optimization parameters too.
        super().__init__(
            source_model=source_model,
            config=config,
            device=device,
        )

        self.beta = float(
            self.config[
                "beta"
            ]
        )

        self.orthogonal_scale = float(
            self.config[
                "orthogonal_scale"
            ]
        )

        self.warmup_updates = int(
            self.config[
                "warmup_updates"
            ]
        )

        self.epsilon = float(
            self.config.get(
                "epsilon",
                1.0e-12,
            )
        )

        if not (
            0.0 <= self.beta < 1.0
        ):
            raise ValueError(
                "temporal_gradient.beta must satisfy "
                "0 <= beta < 1."
            )

        if self.orthogonal_scale < 0.0:
            raise ValueError(
                "temporal_gradient.orthogonal_scale "
                "must be >= 0."
            )

        if self.warmup_updates < 0:
            raise ValueError(
                "temporal_gradient.warmup_updates "
                "must be >= 0."
            )

        if self.epsilon <= 0.0:
            raise ValueError(
                "temporal_gradient.epsilon must be > 0."
            )

        self._reset_temporal_state()

    # ========================================================
    # RESET
    # ========================================================

    def _reset_adaptation_state(
        self,
    ) -> None:
        """
        Restore optimizer, Metadata-TTA diagnostics and
        temporal gradient memory.
        """

        self._configure_trainable_parameters()

        self.optimizer = (
            self._build_optimizer()
        )

        self._reset_method_stats()

        self._reset_temporal_state()

    def _reset_temporal_state(
        self,
    ) -> None:

        self.gradient_ema: (
            list[torch.Tensor]
            | None
        ) = None

        # Counts actual parameter updates / accepted metadata
        # gradients that have contributed to temporal memory.
        self.temporal_update_count = 0

        self.n_warmup_updates = 0
        self.n_temporal_updates = 0

        self.sum_raw_gradient_norm = 0.0
        self.sum_ema_gradient_norm = 0.0
        self.sum_parallel_gradient_norm = 0.0
        self.sum_orthogonal_gradient_norm = 0.0
        self.sum_temporal_gradient_norm = 0.0
        self.sum_gradient_cosine = 0.0

    # ========================================================
    # TENSOR-LIST UTILITIES
    # ========================================================

    @staticmethod
    def _clone_gradient(
        gradient: list[torch.Tensor],
    ) -> list[torch.Tensor]:

        return [
            tensor
            .detach()
            .clone()

            for tensor
            in gradient
        ]

    @staticmethod
    def _dot(
        a: list[torch.Tensor],
        b: list[torch.Tensor],
    ) -> torch.Tensor:

        if len(a) != len(b):
            raise ValueError(
                "Gradient lists must have equal length."
            )

        if not a:
            raise ValueError(
                "Gradient list cannot be empty."
            )

        result = torch.zeros(
            (),
            dtype=torch.float32,
            device=a[0].device,
        )

        for (
            tensor_a,
            tensor_b,
        ) in zip(
            a,
            b,
        ):

            result = (
                result
                + torch.sum(
                    tensor_a.float()
                    * tensor_b.float()
                )
            )

        return result

    def _tensor_list_norm(
        self,
        gradient: list[torch.Tensor],
    ) -> float:

        if not gradient:
            return 0.0

        squared_norm = self._dot(
            gradient,
            gradient,
        )

        return float(
            torch.sqrt(
                torch.clamp(
                    squared_norm,
                    min=0.0,
                )
            ).item()
        )

    def _tensor_list_cosine(
        self,
        a: list[torch.Tensor],
        b: list[torch.Tensor],
    ) -> float:

        norm_a = self._tensor_list_norm(
            a
        )

        norm_b = self._tensor_list_norm(
            b
        )

        denominator = (
            norm_a
            * norm_b
        )

        if denominator <= self.epsilon:
            return 0.0

        dot = float(
            self._dot(
                a,
                b,
            ).item()
        )

        cosine = (
            dot
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

    # ========================================================
    # RAW METADATA GRADIENT
    # ========================================================

    def _compute_raw_metadata_gradient(
        self,
        x_tensor: torch.Tensor,
        y_aux_tensor: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
    ]:
        """
        Compute the auxiliary metadata gradient only.

        Source-anchor regularization is deliberately not
        included here because temporal filtering should operate
        on the metadata signal, not on the anchor.
        """

        self.optimizer.zero_grad(
            set_to_none=True
        )

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

        aux_loss = F.cross_entropy(
            aux_logits,
            y_aux_tensor,
        )

        gradients = torch.autograd.grad(
            aux_loss,
            self.trainable_parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )

        raw_gradient = [
            gradient
            .detach()
            .clone()

            for gradient
            in gradients
        ]

        return (
            aux_loss.detach(),
            raw_gradient,
        )

    # ========================================================
    # TEMPORAL EMA
    # ========================================================

    def _update_gradient_ema(
        self,
        raw_gradient: list[torch.Tensor],
    ) -> None:
        """
        Update temporal memory after processing the current
        gradient.

        Therefore the EMA used to transform sample t contains
        only gradients from samples < t.
        """

        if self.gradient_ema is None:

            self.gradient_ema = (
                self._clone_gradient(
                    raw_gradient
                )
            )

            return

        self.gradient_ema = [
            (
                self.beta
                * old_gradient
                +
                (
                    1.0
                    - self.beta
                )
                * current_gradient
            )
            .detach()
            .clone()

            for (
                old_gradient,
                current_gradient,
            ) in zip(
                self.gradient_ema,
                raw_gradient,
            )
        ]

    # ========================================================
    # TEMPORAL GRADIENT DECOMPOSITION
    # ========================================================

    def _temporal_gradient(
        self,
        raw_gradient: list[torch.Tensor],
    ) -> tuple[
        list[torch.Tensor],
        dict[str, float],
    ]:
        """
        Build the temporally filtered metadata gradient.
        """

        raw_norm = self._tensor_list_norm(
            raw_gradient
        )

        # ----------------------------------------------------
        # WARMUP / NO TEMPORAL DIRECTION
        # ----------------------------------------------------

        if (
            self.gradient_ema is None
            or self.temporal_update_count
            < self.warmup_updates
        ):

            return (
                self._clone_gradient(
                    raw_gradient
                ),
                {
                    "raw_gradient_norm":
                        raw_norm,

                    "ema_gradient_norm":
                        (
                            0.0
                            if self.gradient_ema is None
                            else self._tensor_list_norm(
                                self.gradient_ema
                            )
                        ),

                    "parallel_gradient_norm":
                        raw_norm,

                    "orthogonal_gradient_norm":
                        0.0,

                    "temporal_gradient_norm":
                        raw_norm,

                    "gradient_cosine":
                        0.0,

                    "warmup":
                        1.0,
                },
            )

        ema_gradient = (
            self.gradient_ema
        )

        ema_norm = self._tensor_list_norm(
            ema_gradient
        )

        if ema_norm <= self.epsilon:

            return (
                self._clone_gradient(
                    raw_gradient
                ),
                {
                    "raw_gradient_norm":
                        raw_norm,

                    "ema_gradient_norm":
                        ema_norm,

                    "parallel_gradient_norm":
                        raw_norm,

                    "orthogonal_gradient_norm":
                        0.0,

                    "temporal_gradient_norm":
                        raw_norm,

                    "gradient_cosine":
                        0.0,

                    "warmup":
                        0.0,
                },
            )

        ema_squared_norm = self._dot(
            ema_gradient,
            ema_gradient,
        )

        projection_coefficient = (
            self._dot(
                raw_gradient,
                ema_gradient,
            )
            / torch.clamp(
                ema_squared_norm,
                min=self.epsilon,
            )
        )

        parallel_gradient = [
            projection_coefficient
            * ema_component

            for ema_component
            in ema_gradient
        ]

        orthogonal_gradient = [
            raw_component
            - parallel_component

            for (
                raw_component,
                parallel_component,
            ) in zip(
                raw_gradient,
                parallel_gradient,
            )
        ]

        temporal_gradient = [
            (
                parallel_component
                + self.orthogonal_scale
                * orthogonal_component
            )

            for (
                parallel_component,
                orthogonal_component,
            ) in zip(
                parallel_gradient,
                orthogonal_gradient,
            )
        ]

        parallel_norm = self._tensor_list_norm(
            parallel_gradient
        )

        orthogonal_norm = self._tensor_list_norm(
            orthogonal_gradient
        )

        temporal_norm = self._tensor_list_norm(
            temporal_gradient
        )

        cosine = self._tensor_list_cosine(
            raw_gradient,
            ema_gradient,
        )

        return (
            temporal_gradient,
            {
                "raw_gradient_norm":
                    raw_norm,

                "ema_gradient_norm":
                    ema_norm,

                "parallel_gradient_norm":
                    parallel_norm,

                "orthogonal_gradient_norm":
                    orthogonal_norm,

                "temporal_gradient_norm":
                    temporal_norm,

                "gradient_cosine":
                    cosine,

                "warmup":
                    0.0,
            },
        )

    # ========================================================
    # SOURCE ANCHOR GRADIENT
    # ========================================================

    def _source_anchor_gradients(
        self,
    ) -> list[torch.Tensor]:
        """
        Gradient of:

            regularization *
            ||theta - theta_source||^2

        with respect to adapter parameters.
        """

        return [
            (
                2.0
                * self.regularization
                * (
                    parameter.detach()
                    - source_value
                )
            )

            for (
                parameter,
                source_value,
            ) in zip(
                self.trainable_parameters,
                self.source_parameter_values,
            )
        ]

    # ========================================================
    # APPLY GRADIENT
    # ========================================================

    def _apply_temporal_gradient(
        self,
        temporal_gradient: list[torch.Tensor],
    ) -> tuple[
        float,
        float,
    ]:

        anchor_gradient = (
            self._source_anchor_gradients()
        )

        final_gradient = [
            temporal_component
            + anchor_component

            for (
                temporal_component,
                anchor_component,
            ) in zip(
                temporal_gradient,
                anchor_gradient,
            )
        ]

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

        for (
            parameter,
            gradient,
        ) in zip(
            self.trainable_parameters,
            final_gradient,
        ):

            parameter.grad = (
                gradient
                .detach()
                .clone()
            )

        parameter_gradient_norm = (
            self._gradient_norm()
        )

        if self.gradient_clip > 0.0:

            torch.nn.utils.clip_grad_norm_(
                self.trainable_parameters,
                max_norm=self.gradient_clip,
            )

        self.optimizer.step()

        parameter_delta = (
            self._parameter_delta_norm(
                parameters=(
                    self.trainable_parameters
                ),
                before=parameters_before,
            )
        )

        return (
            float(
                parameter_gradient_norm
            ),
            float(
                parameter_delta
            ),
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
                "TemporalGradientTTA requires "
                "auxiliary labels."
            )

        x_tensor = self._as_feature_tensor(
            x
        )

        y_aux_tensor = self._as_label_tensor(
            y_aux
        )

        if len(x_tensor) != 1:
            raise ValueError(
                "TemporalGradientTTA is defined "
                "sample-by-sample and requires batch size 1."
            )

        if len(y_aux_tensor) != 1:
            raise ValueError(
                "TemporalGradientTTA requires exactly "
                "one auxiliary label per observation."
            )

        if (
            torch.any(
                y_aux_tensor < 0
            )
            or torch.any(
                y_aux_tensor
                >= self.n_classes_aux
            )
        ):

            return AdaptationResult(
                applied=False,
                diagnostics={
                    "reason":
                        "invalid_aux_label",
                },
            )

        # ====================================================
        # AUXILIARY LOSS GATE
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
                },
            )

        self.n_aux_window_pass += 1

        # ====================================================
        # ONE OR MORE ADAPTATION STEPS
        # ====================================================

        final_step_diagnostics: dict[
            str,
            float,
        ] = {}

        parameter_gradient_sum = 0.0
        parameter_delta_sum = 0.0

        for _ in range(
            self.steps
        ):

            (
                aux_loss,
                raw_gradient,
            ) = (
                self._compute_raw_metadata_gradient(
                    x_tensor=x_tensor,
                    y_aux_tensor=(
                        y_aux_tensor
                    ),
                )
            )

            (
                temporal_gradient,
                step_diagnostics,
            ) = (
                self._temporal_gradient(
                    raw_gradient
                )
            )

            if bool(
                step_diagnostics[
                    "warmup"
                ]
            ):
                self.n_warmup_updates += 1
            else:
                self.n_temporal_updates += 1

            (
                parameter_gradient_norm,
                parameter_delta,
            ) = (
                self._apply_temporal_gradient(
                    temporal_gradient
                )
            )

            # The current metadata gradient becomes part of
            # temporal memory only after its update direction
            # has been constructed.
            self._update_gradient_ema(
                raw_gradient
            )

            self.temporal_update_count += 1

            self.sum_raw_gradient_norm += (
                step_diagnostics[
                    "raw_gradient_norm"
                ]
            )

            self.sum_ema_gradient_norm += (
                step_diagnostics[
                    "ema_gradient_norm"
                ]
            )

            self.sum_parallel_gradient_norm += (
                step_diagnostics[
                    "parallel_gradient_norm"
                ]
            )

            self.sum_orthogonal_gradient_norm += (
                step_diagnostics[
                    "orthogonal_gradient_norm"
                ]
            )

            self.sum_temporal_gradient_norm += (
                step_diagnostics[
                    "temporal_gradient_norm"
                ]
            )

            self.sum_gradient_cosine += (
                step_diagnostics[
                    "gradient_cosine"
                ]
            )

            parameter_gradient_sum += (
                parameter_gradient_norm
            )

            parameter_delta_sum += (
                parameter_delta
            )

            final_step_diagnostics = (
                step_diagnostics
            )

            final_step_diagnostics[
                "aux_loss"
            ] = float(
                aux_loss.item()
            )

        self._record_update()

        mean_parameter_gradient_norm = (
            parameter_gradient_sum
            / float(
                self.steps
            )
        )

        mean_parameter_delta = (
            parameter_delta_sum
            / float(
                self.steps
            )
        )

        self.sum_gradient_norm += (
            mean_parameter_gradient_norm
        )

        self.sum_parameter_delta += (
            mean_parameter_delta
        )

        return AdaptationResult(
            applied=True,
            diagnostics={
                "reason":
                    (
                        "warmup_update"
                        if bool(
                            final_step_diagnostics[
                                "warmup"
                            ]
                        )
                        else "temporal_update"
                    ),

                "aux_loss":
                    float(
                        final_step_diagnostics[
                            "aux_loss"
                        ]
                    ),

                "normalized_aux_loss":
                    normalized_aux_loss,

                "warmup":
                    bool(
                        final_step_diagnostics[
                            "warmup"
                        ]
                    ),

                "raw_gradient_norm":
                    float(
                        final_step_diagnostics[
                            "raw_gradient_norm"
                        ]
                    ),

                "ema_gradient_norm":
                    float(
                        final_step_diagnostics[
                            "ema_gradient_norm"
                        ]
                    ),

                "parallel_gradient_norm":
                    float(
                        final_step_diagnostics[
                            "parallel_gradient_norm"
                        ]
                    ),

                "orthogonal_gradient_norm":
                    float(
                        final_step_diagnostics[
                            "orthogonal_gradient_norm"
                        ]
                    ),

                "temporal_gradient_norm":
                    float(
                        final_step_diagnostics[
                            "temporal_gradient_norm"
                        ]
                    ),

                "gradient_cosine":
                    float(
                        final_step_diagnostics[
                            "gradient_cosine"
                        ]
                    ),

                "parameter_gradient_norm":
                    mean_parameter_gradient_norm,

                "parameter_delta":
                    mean_parameter_delta,
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

        n_temporal_steps = max(
            self.temporal_update_count,
            1,
        )

        diagnostics.update(
            {
                "warmup_update_count":
                    int(
                        self.n_warmup_updates
                    ),

                "temporal_update_count":
                    int(
                        self.n_temporal_updates
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

                "mean_raw_gradient_norm":
                    float(
                        self.sum_raw_gradient_norm
                        / n_temporal_steps
                    ),

                "mean_ema_gradient_norm":
                    float(
                        self.sum_ema_gradient_norm
                        / n_temporal_steps
                    ),

                "mean_parallel_gradient_norm":
                    float(
                        self.sum_parallel_gradient_norm
                        / n_temporal_steps
                    ),

                "mean_orthogonal_gradient_norm":
                    float(
                        self.sum_orthogonal_gradient_norm
                        / n_temporal_steps
                    ),

                "mean_temporal_gradient_norm":
                    float(
                        self.sum_temporal_gradient_norm
                        / n_temporal_steps
                    ),

                "mean_gradient_cosine":
                    float(
                        self.sum_gradient_cosine
                        / n_temporal_steps
                    ),

                "mean_parameter_gradient_norm":
                    float(
                        self.sum_gradient_norm
                        / n_updates
                    ),

                "mean_parameter_delta":
                    float(
                        self.sum_parameter_delta
                        / n_updates
                    ),
            }
        )

        return diagnostics