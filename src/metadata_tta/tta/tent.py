from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .base import (
    AdaptationResult,
    EvaluationOrder,
    TTAMethod,
)


class TentTTA(TTAMethod):
    """
    Test-time entropy minimization (TENT).

    Stable V2 semantics:
    - start from the SingleHead source model;
    - do not use metadata labels;
    - do not use main-task labels;
    - adapt only affine BatchNorm parameters;
    - use target-batch BatchNorm statistics;
    - keep stochastic layers such as Dropout disabled.

    The evaluator must enforce the strict prequential order:

        predict current batch
        record predictions
        observe current batch
        update

    No metadata labels and no main-task labels are used.
    """

    method_name = "tent"

    requires_aux_labels = False

    evaluation_order = EvaluationOrder.PREDICT_THEN_ADAPT

    stream_batch_size = 64
    minimum_stream_batch_size = 2

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

        self.learning_rate = float(
            self.config["learning_rate"]
        )
        self.weight_decay = float(
            self.config.get(
                "weight_decay",
                0.0,
            )
        )

        self.optimizer_name = str(
            self.config.get(
                "optimizer",
                "adam",
            )
        ).lower()

        self.batch_size = int(
            self.config.get(
                "batch_size",
                64,
            )
        )
        self.stream_batch_size = self.batch_size
        self.steps = int(
            self.config.get(
                "steps",
                1,
            )
        )

        if self.learning_rate <= 0.0:
            raise ValueError(
                "tent.learning_rate must be > 0."
            )

        if self.weight_decay < 0.0:
            raise ValueError(
                "tent.weight_decay must be >= 0."
            )

        if self.batch_size < 2:
            raise ValueError(
                "TENT requires batch_size >= 2 "
                "because BatchNorm uses target-batch "
                "statistics."
            )

        if self.steps < 1:
            raise ValueError(
                "tent.steps must be >= 1."
            )

        if self.optimizer_name != "adam":
            raise ValueError(
                "The stable V2 TENT implementation "
                "currently supports optimizer='adam' only."
            )

        self._configure_model()

        self.optimizer = (
            self._build_optimizer()
        )

        self.sum_entropy = 0.0
        self.n_adapted_samples = 0

    # ========================================================
    # MODEL CONFIGURATION
    # ========================================================

    @staticmethod
    def _is_batch_norm(
        module: nn.Module,
    ) -> bool:
        return isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
            ),
        )

    def _set_tent_mode(
        self,
    ) -> None:
        """
        Keep stochastic layers disabled while letting BatchNorm
        layers use current target-batch statistics.

        Important:
        - model.eval() disables Dropout;
        - BatchNorm modules are then put in train mode;
        - track_running_stats=False makes BatchNorm use batch
          statistics without deleting running_mean/running_var.

        We deliberately do NOT set:

            module.running_mean = None
            module.running_var = None

        because removing those registered buffers breaks exact
        source-state restoration through load_state_dict(strict=True).
        """

        self.model.eval()

        for module in self.model.modules():
            if self._is_batch_norm(
                module
            ):
                module.train()
                module.track_running_stats = False

    def _configure_model(
        self,
    ) -> None:
        """
        Configure the model following TENT semantics.

        - freeze every parameter;
        - enable only BatchNorm affine weight/bias;
        - use target-batch statistics at test time;
        - keep running-stat buffers registered for exact reset;
        - keep Dropout and other stochastic layers disabled.
        """

        self._set_tent_mode()

        for parameter in self.model.parameters():
            parameter.requires_grad_(
                False
            )

        trainable_parameters: list[
            nn.Parameter
        ] = []

        batch_norm_count = 0

        for module in self.model.modules():

            if self._is_batch_norm(
                module
            ):
                batch_norm_count += 1

                module.train()
                module.track_running_stats = False

                if module.affine:

                    if module.weight is not None:
                        module.weight.requires_grad_(
                            True
                        )

                        trainable_parameters.append(
                            module.weight
                        )

                    if module.bias is not None:
                        module.bias.requires_grad_(
                            True
                        )

                        trainable_parameters.append(
                            module.bias
                        )

        if batch_norm_count == 0:
            raise RuntimeError(
                "TentTTA requires at least one "
                "BatchNorm layer."
            )

        if not trainable_parameters:
            raise RuntimeError(
                "TentTTA found no trainable "
                "BatchNorm affine parameters."
            )

        self.trainable_parameters = (
            trainable_parameters
        )

    def _build_optimizer(
        self,
    ) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.trainable_parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    # ========================================================
    # RESET
    # ========================================================

    def _reset_adaptation_state(
        self,
    ) -> None:

        self._configure_model()

        self.optimizer = (
            self._build_optimizer()
        )
        self.sum_entropy = 0.0
        self.n_adapted_samples = 0

    # ========================================================
    # ENTROPY
    # ========================================================

    @staticmethod
    def _entropy(
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean predictive entropy over the batch.
        """

        log_probabilities = (
            torch.log_softmax(
                logits,
                dim=1,
            )
        )
        probabilities = (
            torch.softmax(
                logits,
                dim=1,
            )
        )

        sample_entropy = -torch.sum(
            probabilities
            * log_probabilities,
            dim=1,
        )

        return sample_entropy.mean()

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict_logits(
        self,
        x: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict BEFORE adapting on this batch.

        No optimizer step occurs here.
        """

        x_tensor = (
            self._as_feature_tensor(
                x
            )
        )

        if len(x_tensor) < 2:
            raise ValueError(
                "TentTTA prediction requires at least "
                "2 samples per batch."
            )

        self._set_tent_mode()

        with torch.no_grad():

            logits = self.model(
                x_tensor
            )

        return logits.detach()

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
        """
        Adapt using prediction entropy only.

        y_aux is intentionally ignored.
        """

        x_tensor = (
            self._as_feature_tensor(
                x
            )
        )

        if len(x_tensor) < 2:
            raise ValueError(
                "TentTTA update requires at least "
                "2 samples per batch."
            )

        self._record_observation()

        self._set_tent_mode()

        entropy_sum = 0.0

        for _ in range(
            self.steps
        ):

            self.optimizer.zero_grad(
                set_to_none=True
            )

            logits = self.model(
                x_tensor
            )
            entropy = self._entropy(
                logits
            )

            entropy.backward()

            self.optimizer.step()

            entropy_sum += float(
                entropy
                .detach()
                .item()
            )

        mean_entropy = (
            entropy_sum
            / float(
                self.steps
            )
        )

        self.sum_entropy += (
            mean_entropy
            * len(x_tensor)
        )
        self.n_adapted_samples += int(
            len(x_tensor)
        )

        self._record_update()

        return AdaptationResult(
            applied=True,
            diagnostics={
                "reason":
                    "entropy_update",

                "entropy":
                    mean_entropy,

                "batch_size":
                    int(
                        len(x_tensor)
                    ),
            },
        )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    def diagnostics(
        self,
    ) -> dict[str, Any]:

        diagnostics = (
            super().diagnostics()
        )

        mean_entropy = (
            self.sum_entropy
            / max(
                self.n_adapted_samples,
                1,
            )
        )

        diagnostics.update(
            {
                "n_adapted_samples":
                    int(
                        self.n_adapted_samples
                    ),

                "mean_entropy":
                    float(
                        mean_entropy
                    ),

                "batch_size":
                    int(
                        self.batch_size
                    ),

                "steps":
                    int(
                        self.steps
                    ),
            }
        )

        return diagnostics