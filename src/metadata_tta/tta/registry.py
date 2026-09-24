from __future__ import annotations

from typing import Type

from .base import TTAMethod
from .metadata_variants import (
    MetadataCumulativeAnnualResetTTA,
    MetadataCumulativeDriftAnnualResetTTA,
    MetadataCumulativeDriftResetTTA,
    MetadataCumulativeTTA,
    MetadataEpisodicTTA,
)
from .temporal_gradient import TemporalGradientEMATTA
from .tent import TentTTA


METHOD_REGISTRY: dict[
    str,
    Type[TTAMethod],
] = {
    "metadata_episodic":
        MetadataEpisodicTTA,

    "metadata_cumulative":
        MetadataCumulativeTTA,

    "metadata_cumulative_annual_reset":
        MetadataCumulativeAnnualResetTTA,

    "metadata_cumulative_drift_reset":
        MetadataCumulativeDriftResetTTA,

    "metadata_cumulative_drift_annual_reset":
        MetadataCumulativeDriftAnnualResetTTA,

    "temporal_gradient_ema":
        TemporalGradientEMATTA,

    "tent":
        TentTTA,
}


def registered_methods() -> tuple[str, ...]:
    return tuple(
        sorted(
            METHOD_REGISTRY.keys()
        )
    )


def method_is_registered(
    name: str,
) -> bool:

    return (
        str(name).strip()
        in METHOD_REGISTRY
    )


def get_method_class(
    name: str,
) -> Type[TTAMethod]:

    normalized_name = str(
        name
    ).strip()

    if normalized_name not in METHOD_REGISTRY:

        available = ", ".join(
            registered_methods()
        )

        raise KeyError(
            f"Unknown TTA method "
            f"{normalized_name!r}. "
            f"Registered methods: {available}"
        )

    return METHOD_REGISTRY[
        normalized_name
    ]