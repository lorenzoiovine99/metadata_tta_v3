from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationRecord:
    year: int
    method: str
    accuracy: float
    n_samples: int


@dataclass(frozen=True)
class EvaluationResult:
    records: tuple[EvaluationRecord, ...]
    diagnostics: dict[str, Any]