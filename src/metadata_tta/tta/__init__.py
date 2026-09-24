from .base import (
    AdaptationResult,
    EvaluationOrder,
    TTAMethod,
)

from .drift import (
    ADWINDriftDetector,
    DriftDetectionResult,
)

from .metadata import (
    MetadataTTA,
)

from .metadata_variants import (
    MetadataCumulativeAnnualResetTTA,
    MetadataCumulativeDriftAnnualResetTTA,
    MetadataCumulativeDriftResetTTA,
    MetadataCumulativeTTA,
    MetadataEpisodicTTA,
)

from .temporal_gradient import (
    TemporalGradientEMATTA,
)

from .tent import (
    TentTTA,
)

from .registry import (
    METHOD_REGISTRY,
    get_method_class,
    method_is_registered,
    registered_methods,
)


__all__ = [
    "AdaptationResult",
    "EvaluationOrder",
    "TTAMethod",
    "ADWINDriftDetector",
    "DriftDetectionResult",
    "MetadataTTA",
    "MetadataEpisodicTTA",
    "MetadataCumulativeTTA",
    "MetadataCumulativeAnnualResetTTA",
    "MetadataCumulativeDriftResetTTA",
    "MetadataCumulativeDriftAnnualResetTTA",
    "TemporalGradientEMATTA",
    "TentTTA",
    "METHOD_REGISTRY",
    "registered_methods",
    "method_is_registered",
    "get_method_class",
]