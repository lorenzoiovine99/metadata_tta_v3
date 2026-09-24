from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .base import AdaptationResult
from .drift import ADWINDriftDetector
from .metadata import MetadataTTA


class MetadataEpisodicTTA(MetadataTTA):
    """
    Episodic metadata-guided TTA.

    For every sample:

        reset adapter to source
        compute metadata loss
        adapt on current sample
        predict current sample afterwards

    The evaluator is responsible only for enforcing
    ADAPT_THEN_PREDICT. The episodic reset belongs here.
    """

    method_name = "metadata_episodic"

    def observe(
        self,
        x: np.ndarray | torch.Tensor,
        y_aux: np.ndarray | torch.Tensor | int | None = None,
    ) -> AdaptationResult:

        self.reset_adapted_parameters_to_source()

        return super().observe(
            x=x,
            y_aux=y_aux,
        )


class MetadataCumulativeTTA(MetadataTTA):
    """
    Plain cumulative metadata-guided TTA.

    Adapter state and optimizer state persist through the
    complete chronological stream.
    """

    method_name = "metadata_cumulative"


class MetadataCumulativeAnnualResetTTA(
    MetadataCumulativeTTA
):
    """
    Cumulative metadata TTA with source reset at year
    boundaries.

    The first year does not count as a reset because the
    method already starts from the source state.
    """

    method_name = "metadata_cumulative_annual_reset"

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

        self._current_year: int | None = None
        self.n_annual_resets = 0

    def on_year_start(
        self,
        year: int,
    ) -> dict[str, Any]:

        year = int(year)

        if self._current_year is None:
            self._current_year = year

            return {
                "year": year,
                "reset_triggered": False,
                "reset_reason": None,
            }

        if year == self._current_year:
            return {
                "year": year,
                "reset_triggered": False,
                "reset_reason": None,
            }

        previous_year = self._current_year
        self._current_year = year

        self.reset_adapted_parameters_to_source()
        self.n_annual_resets += 1

        return {
            "year": year,
            "previous_year": previous_year,
            "reset_triggered": True,
            "reset_reason": "annual",
        }

    def _reset_adaptation_state(
        self,
    ) -> None:

        super()._reset_adaptation_state()

        self._current_year = None
        self.n_annual_resets = 0

    def diagnostics(
        self,
    ) -> dict[str, Any]:

        diagnostics = super().diagnostics()

        diagnostics.update(
            {
                "annual_reset_count":
                    int(self.n_annual_resets),
            }
        )

        return diagnostics


class _MetadataDriftMixin:
    """
    ADWIN reset policy shared by the two drift variants.

    Scientific order for sample t:

        1. compute pre-update normalized auxiliary loss;
        2. feed it to ADWIN;
        3. if drift is detected, reset adapter to SOURCE;
        4. adapt using sample t;
        5. evaluator predicts sample t post-update.

    Therefore the sample that triggers drift becomes the first
    adaptation sample of the new regime.
    """

    drift_signal = "normalized_aux_loss"

    def _initialize_drift_detector(
        self,
    ) -> None:

        configured_signal = str(
            self.config.get(
                "drift_signal",
                "normalized_aux_loss",
            )
        )

        if configured_signal != "normalized_aux_loss":
            raise ValueError(
                "Metadata drift-reset V3 currently supports "
                "drift_signal='normalized_aux_loss' only."
            )

        self.adwin_delta = float(
            self.config["adwin_delta"]
        )

        self.drift_detector = ADWINDriftDetector(
            delta=self.adwin_delta
        )

        self.n_drift_resets = 0
        self.drift_events: list[
            dict[str, Any]
        ] = []

    def _pre_update_drift_check(
        self,
        x: np.ndarray | torch.Tensor,
        y_aux: np.ndarray | torch.Tensor | int | None,
    ) -> dict[str, Any]:

        if y_aux is None:
            raise ValueError(
                f"{self.method_name} requires auxiliary labels."
            )

        x_tensor = self._as_feature_tensor(x)
        y_aux_tensor = self._as_label_tensor(y_aux)

        if len(x_tensor) != 1:
            raise ValueError(
                f"{self.method_name} is defined sample-by-sample "
                "and requires batch size 1."
            )

        if len(y_aux_tensor) != 1:
            raise ValueError(
                f"{self.method_name} requires exactly one "
                "auxiliary label per sample."
            )

        if (
            torch.any(y_aux_tensor < 0)
            or torch.any(
                y_aux_tensor >= self.n_classes_aux
            )
        ):
            return {
                "drift_checked": False,
                "drift_detected": False,
                "drift_signal":
                    self.drift_signal,
                "drift_signal_value": None,
            }

        self.model.eval()

        with torch.no_grad():
            aux_loss = self._auxiliary_loss(
                x_tensor=x_tensor,
                y_aux_tensor=y_aux_tensor,
            )

        aux_loss_value = float(
            aux_loss.item()
        )

        normalized_aux_loss = float(
            aux_loss_value
            / max(
                self.aux_loss_normalizer,
                self.epsilon,
            )
        )

        detector_result = (
            self.drift_detector.observe(
                normalized_aux_loss
            )
        )

        diagnostics: dict[str, Any] = {
            "drift_checked": True,
            "drift_detected":
                bool(
                    detector_result.drift_detected
                ),
            "drift_signal":
                self.drift_signal,
            "drift_signal_value":
                normalized_aux_loss,
            "pre_update_aux_loss":
                aux_loss_value,
            "pre_update_normalized_aux_loss":
                normalized_aux_loss,
        }

        if detector_result.drift_detected:

            self.reset_adapted_parameters_to_source()

            # Start a fresh ADWIN window for the new regime.
            self.drift_detector.reset_window()

            self.n_drift_resets += 1

            event = {
                "observation_index":
                    int(
                        self.number_of_observations
                    ),
                "normalized_aux_loss":
                    normalized_aux_loss,
                "adwin_delta":
                    self.adwin_delta,
            }

            self.drift_events.append(event)

            diagnostics.update(
                {
                    "reset_triggered": True,
                    "reset_reason": "drift",
                    "drift_reset_index":
                        int(
                            self.n_drift_resets
                        ),
                }
            )

        else:

            diagnostics.update(
                {
                    "reset_triggered": False,
                    "reset_reason": None,
                }
            )

        return diagnostics

    def record_reset_location(
        self,
        *,
        year: int,
        sample_index: int,
        global_sample_index: int,
        diagnostics: Mapping[str, Any],
    ) -> None:
        """
        Enrich the most recently detected drift event with its
        temporal stream location.

        This method is called by the evaluator only AFTER the
        adaptation decision has already been made.

        Therefore year/sample indices are diagnostic metadata
        only and cannot influence ADWIN or adaptation.
        """

        if not bool(
            diagnostics.get(
                "reset_triggered",
                False,
            )
        ):
            return

        if (
            diagnostics.get(
                "reset_reason"
            )
            != "drift"
        ):
            return

        if not self.drift_events:
            raise RuntimeError(
                "A drift reset was reported but no "
                "drift event exists to annotate."
            )

        event_number = int(
            diagnostics.get(
                "drift_reset_index",
                self.n_drift_resets,
            )
        )

        if event_number != len(
            self.drift_events
        ):
            raise RuntimeError(
                "Drift event bookkeeping mismatch: "
                f"event_number={event_number}, "
                f"stored_events={len(self.drift_events)}."
            )

        event = self.drift_events[-1]

        event.update(
            {
                "drift_event_number":
                    event_number,

                "year":
                    int(year),

                "year_sample_index":
                    int(sample_index),

                "global_sample_index":
                    int(global_sample_index),

                "pre_update_aux_loss":
                    (
                        None
                        if diagnostics.get(
                            "pre_update_aux_loss"
                        ) is None
                        else float(
                            diagnostics[
                                "pre_update_aux_loss"
                            ]
                        )
                    ),

                "normalized_aux_loss":
                    (
                        None
                        if diagnostics.get(
                            "pre_update_normalized_aux_loss"
                        ) is None
                        else float(
                            diagnostics[
                                "pre_update_normalized_aux_loss"
                            ]
                        )
                    ),

                "adwin_delta":
                    float(self.adwin_delta),
            }
        )

    def observe(
        self,
        x: np.ndarray | torch.Tensor,
        y_aux: np.ndarray | torch.Tensor | int | None = None,
    ) -> AdaptationResult:

        drift_diagnostics = (
            self._pre_update_drift_check(
                x=x,
                y_aux=y_aux,
            )
        )

        adaptation_result = super().observe(
            x=x,
            y_aux=y_aux,
        )

        diagnostics = dict(
            adaptation_result.diagnostics
        )

        diagnostics.update(
            drift_diagnostics
        )

        return AdaptationResult(
            applied=adaptation_result.applied,
            diagnostics=diagnostics,
        )

    def diagnostics(
        self,
    ) -> dict[str, Any]:

        diagnostics = super().diagnostics()

        diagnostics.update(
            {
                "drift_signal":
                    self.drift_signal,

                "adwin_delta":
                    float(self.adwin_delta),

                "drift_reset_count":
                    int(self.n_drift_resets),

                "drift_detector_observations":
                    int(
                        self.drift_detector.n_observations
                    ),

                "drift_detector_detections":
                    int(
                        self.drift_detector.n_detections
                    ),

                "drift_events":
                    list(self.drift_events),
            }
        )

        return diagnostics


class MetadataCumulativeDriftResetTTA(
    _MetadataDriftMixin,
    MetadataCumulativeTTA,
):
    """
    Cumulative metadata TTA with ADWIN drift reset.
    """

    method_name = "metadata_cumulative_drift_reset"

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

        self._initialize_drift_detector()

    def _reset_adaptation_state(
        self,
    ) -> None:

        super()._reset_adaptation_state()

        if hasattr(
            self,
            "drift_detector",
        ):
            self.drift_detector.reset(
                reset_statistics=True
            )

            self.n_drift_resets = 0
            self.drift_events = []


class MetadataCumulativeDriftAnnualResetTTA(
    _MetadataDriftMixin,
    MetadataCumulativeAnnualResetTTA,
):
    """
    Cumulative metadata TTA with both:

        - ADWIN drift resets;
        - source reset at year boundaries.
    """

    method_name = (
        "metadata_cumulative_drift_annual_reset"
    )

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

        self._initialize_drift_detector()

    def on_year_start(
        self,
        year: int,
    ) -> dict[str, Any]:

        event = super().on_year_start(
            year=year
        )

        if bool(
            event.get(
                "reset_triggered",
                False,
            )
        ):
            # Annual reset starts a new regime for both the
            # adapter and the drift detector.
            self.drift_detector.reset_window()

        return event

    def _reset_adaptation_state(
        self,
    ) -> None:

        super()._reset_adaptation_state()

        if hasattr(
            self,
            "drift_detector",
        ):
            self.drift_detector.reset(
                reset_statistics=True
            )

            self.n_drift_resets = 0
            self.drift_events = []