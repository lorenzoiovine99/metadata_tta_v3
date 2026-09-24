from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


class EvaluationOrder(str, Enum):
    """
    Defines when the main-task prediction used for evaluation
    is produced relative to the adaptation step.
    """

    ADAPT_THEN_PREDICT = "adapt_then_predict"
    PREDICT_THEN_ADAPT = "predict_then_adapt"


@dataclass(frozen=True)
class AdaptationResult:
    """
    Result of one adaptation observation.

    applied
        True only when model parameters were actually updated.

    diagnostics
        Optional scalar/event diagnostics for this observation.
    """

    applied: bool
    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


class TTAMethod(ABC):
    """
    Base interface for test-time adaptation methods.

    Main-task labels are deliberately absent from every
    adaptation interface.

    Each method explicitly declares its evaluation order:

        ADAPT_THEN_PREDICT
            Used by metadata-supervised TTA.

            For the current sample x_t:
                observe/update using metadata_t
                predict x_t with the resulting state
                score that prediction

        PREDICT_THEN_ADAPT
            Used by methods such as TENT.

            For the current batch:
                predict with current state
                score that prediction
                adapt using the same unlabeled batch

            The update therefore affects future batches only.
    """

    method_name: str = "base"
    requires_aux_labels: bool = False

    evaluation_order: EvaluationOrder

    def __init__(
        self,
        source_model: nn.Module,
        config: Mapping[str, Any],
        device: torch.device,
    ) -> None:

        if not isinstance(source_model, nn.Module):
            raise TypeError(
                "source_model must be a torch.nn.Module."
            )

        if not isinstance(config, Mapping):
            raise TypeError(
                "config must be a mapping."
            )

        if not hasattr(
            self.__class__,
            "evaluation_order",
        ):
            raise TypeError(
                f"{self.__class__.__name__} must declare "
                "evaluation_order."
            )

        if not isinstance(
            self.evaluation_order,
            EvaluationOrder,
        ):
            raise TypeError(
                f"{self.__class__.__name__}.evaluation_order "
                "must be an EvaluationOrder."
            )

        self.device = device

        self.config: dict[str, Any] = copy.deepcopy(
            dict(config)
        )

        # Every TTA method owns an independent model copy.
        # The source model supplied by the training pipeline
        # must never be modified by adaptation.
        self.model = copy.deepcopy(
            source_model
        ).to(
            self.device
        )

        # Immutable numerical reference to the complete source
        # checkpoint, including parameters and buffers.
        #
        # Kept on CPU to avoid unnecessarily duplicating the
        # full source checkpoint on accelerator memory.
        self._source_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor
            in self.model.state_dict().items()
        }

        self._number_of_observations = 0
        self._number_of_updates = 0

    @abstractmethod
    def predict_logits(
        self,
        x: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """
        Return main-task logits from the current model state.

        This operation MUST NOT adapt the model.
        """

        raise NotImplementedError

    @abstractmethod
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
        Observe test-time information and optionally adapt.

        Main-task labels are intentionally not accepted.

        Metadata methods receive y_aux.
        Unsupervised methods such as TENT ignore y_aux.
        """

        raise NotImplementedError

    def on_year_start(
        self,
        year: int,
    ) -> dict[str, Any]:
        """
        Optional hook called exactly once before evaluating a
        new stream year.

        Stateful methods may override it to implement an annual
        reset.

        The default implementation performs no action.
        """

        return {
            "year": int(year),
            "reset_triggered": False,
        }

    def reset(
        self,
    ) -> None:
        """
        Fully restore this method to its original source state.

        This is a full experiment-level reset, not necessarily
        the same operation as an episodic, annual, or
        drift-triggered adapter reset.
        """

        self._restore_source_model()

        self._number_of_observations = 0
        self._number_of_updates = 0

        self._reset_adaptation_state()

    def _restore_source_model(
        self,
    ) -> None:
        """
        Restore all parameters and buffers from the immutable
        source checkpoint.
        """

        state = {
            name: tensor.clone()
            for name, tensor
            in self._source_state.items()
        }

        self.model.load_state_dict(
            state,
            strict=True,
        )

        self.model.to(
            self.device
        )

    def _reset_adaptation_state(
        self,
    ) -> None:
        """
        Optional subclass hook after a full source reset.

        Typical uses:
            rebuild optimizer;
            clear gradient EMA;
            reset drift detector;
            clear method diagnostics.
        """

        return None

    def _as_feature_tensor(
        self,
        x: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert features to a float32 2-D tensor on the active
        device.

        A single 1-D feature vector becomes a batch of size 1.
        """

        if isinstance(x, torch.Tensor):
            tensor = x.to(
                device=self.device,
                dtype=torch.float32,
            )
        else:
            tensor = torch.as_tensor(
                x,
                dtype=torch.float32,
                device=self.device,
            )

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        if tensor.ndim != 2:
            raise ValueError(
                "TTA feature input must be 1-D or 2-D. "
                f"Received shape {tuple(tensor.shape)}."
            )

        return tensor

    def _as_label_tensor(
        self,
        y: (
            np.ndarray
            | torch.Tensor
            | int
        ),
    ) -> torch.Tensor:
        """
        Convert labels to a one-dimensional LongTensor on the
        active device.
        """

        if isinstance(y, torch.Tensor):
            tensor = y.to(
                device=self.device,
                dtype=torch.long,
            )
        else:
            tensor = torch.as_tensor(
                y,
                dtype=torch.long,
                device=self.device,
            )

        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)

        return tensor.reshape(-1)

    def source_tensor(
        self,
        name: str,
    ) -> torch.Tensor:
        """
        Return one tensor from the immutable source checkpoint
        on the active device.
        """

        if name not in self._source_state:
            raise KeyError(
                f"Unknown source-state tensor: {name}"
            )

        return self._source_state[
            name
        ].to(
            self.device
        )

    def _record_observation(
        self,
    ) -> None:
        self._number_of_observations += 1

    def _record_update(
        self,
    ) -> None:
        self._number_of_updates += 1

    @property
    def number_of_observations(
        self,
    ) -> int:
        return int(
            self._number_of_observations
        )

    @property
    def number_of_updates(
        self,
    ) -> int:
        return int(
            self._number_of_updates
        )

    @property
    def update_rate(
        self,
    ) -> float:

        if self._number_of_observations == 0:
            return 0.0

        return float(
            self._number_of_updates
            / self._number_of_observations
        )

    def predict_classes(
        self,
        x: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """
        Convenience prediction without adaptation.
        """

        logits = self.predict_logits(x)

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        return (
            predictions.detach()
            .cpu()
            .numpy()
            .astype(
                np.int64,
                copy=False,
            )
        )

    def diagnostics(
        self,
    ) -> dict[str, Any]:
        """
        Common cumulative diagnostics.
        """

        return {
            "method":
                self.method_name,

            "evaluation_order":
                self.evaluation_order.value,

            "n_observations":
                self.number_of_observations,

            "n_updates":
                self.number_of_updates,

            "update_rate":
                self.update_rate,
        }