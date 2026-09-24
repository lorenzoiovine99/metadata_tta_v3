from __future__ import annotations

from dataclasses import dataclass

from capymoa.drift.detectors import ADWIN


@dataclass(frozen=True)
class DriftDetectionResult:
    """
    Result of one drift-detector observation.
    """

    value: float
    drift_detected: bool


class ADWINDriftDetector:
    """
    Small wrapper around CapyMOA ADWIN.

    The detector is deliberately independent from the TTA
    method. It receives only a scalar signal and has no access
    to model parameters, auxiliary labels, main labels, years,
    or reset logic.

    The caller decides:
    - which signal is observed;
    - what to reset after a detected drift;
    - how reset events are logged.
    """

    def __init__(
        self,
        *,
        delta: float,
    ) -> None:

        self.delta = float(delta)

        if not 0.0 < self.delta < 1.0:
            raise ValueError(
                "ADWIN delta must satisfy 0 < delta < 1. "
                f"Received {self.delta}."
            )

        self._detector = self._build_detector()
        self._n_observations = 0
        self._n_detections = 0

    def _build_detector(self) -> ADWIN:
        return ADWIN(
            delta=self.delta,
        )

    @property
    def n_observations(self) -> int:
        return self._n_observations

    @property
    def n_detections(self) -> int:
        return self._n_detections

    def observe(
        self,
        value: float,
    ) -> DriftDetectionResult:
        """
        Feed one scalar observation to ADWIN.
        """

        value = float(value)

        self._detector.add_element(value)

        self._n_observations += 1

        drift_detected = bool(
            self._detector.detected_change()
        )

        if drift_detected:
            self._n_detections += 1

        return DriftDetectionResult(
            value=value,
            drift_detected=drift_detected,
        )

    def reset_window(self) -> None:
        """
        Start a fresh ADWIN window.

        Lifetime diagnostic counters are intentionally preserved.
        """

        self._detector = self._build_detector()

    def reset(
        self,
        *,
        reset_statistics: bool = True,
    ) -> None:
        """
        Fully reset the detector.

        Parameters
        ----------
        reset_statistics:
            If True, also clear lifetime counters.
        """

        self._detector = self._build_detector()

        if reset_statistics:
            self._n_observations = 0
            self._n_detections = 0