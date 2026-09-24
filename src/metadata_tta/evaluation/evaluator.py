from __future__ import annotations

import numpy as np
import torch
from torch import nn

from metadata_tta.tta import (
    EvaluationOrder,
    TTAMethod,
)

from .metrics import accuracy
from .records import (
    EvaluationRecord,
    EvaluationResult,
)


def _as_numpy(
    value: np.ndarray | torch.Tensor,
) -> np.ndarray:

    if isinstance(
        value,
        torch.Tensor,
    ):
        return (
            value
            .detach()
            .cpu()
            .numpy()
        )

    return np.asarray(
        value
    )


def _predict_frozen(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:

    model.eval()

    x_tensor = torch.as_tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        logits = model(
            x_tensor
        )

    return (
        logits.argmax(
            dim=1
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )


def evaluate_frozen_year(
    model: nn.Module,
    X: np.ndarray,
    y_main: np.ndarray,
    year: int,
    method_name: str,
    device: torch.device,
) -> EvaluationRecord:

    predictions = _predict_frozen(
        model=model,
        X=X,
        device=device,
    )

    return EvaluationRecord(
        year=int(year),
        method=str(method_name),
        accuracy=accuracy(
            y_main,
            predictions,
        ),
        n_samples=int(
            len(y_main)
        ),
    )


def _prediction_from_single_logits(
    logits: torch.Tensor,
) -> int:

    if logits.ndim != 2:
        raise RuntimeError(
            "Expected logits with shape "
            "[batch, classes]."
        )

    if len(logits) != 1:
        raise RuntimeError(
            "Sample-wise evaluation expected "
            "exactly one prediction."
        )

    return int(
        logits.argmax(
            dim=1
        )[0].item()
    )

def _samplewise_predictions(
    method: TTAMethod,
    X: np.ndarray,
    y_aux: np.ndarray | None,
    *,
    year: int,
    global_sample_offset: int,
) -> np.ndarray:
    """
    Generic sample-wise evaluator.

    Scientific ordering is declared by the method through
    EvaluationOrder rather than inferred from method_name.

    ADAPT_THEN_PREDICT:
        observe/update x_t
        optionally record reset location
        predict x_t after adaptation

    PREDICT_THEN_ADAPT:
        predict x_t
        observe/update x_t afterwards
        optionally record reset location

    y_main is deliberately unavailable here.

    year and global_sample_offset are diagnostic context only.
    They never affect adaptation decisions.
    """

    predictions: list[int] = []

    for index in range(
        len(X)
    ):

        x = X[index]

        auxiliary_label = (
            None
            if y_aux is None
            else int(
                y_aux[index]
            )
        )

        adaptation_result = None

        if (
            method.evaluation_order
            is EvaluationOrder.ADAPT_THEN_PREDICT
        ):

            adaptation_result = method.observe(
                x=x,
                y_aux=auxiliary_label,
            )

            # ------------------------------------------------
            # Optional diagnostic recording.
            #
            # This happens AFTER observe(), therefore it cannot
            # influence drift detection or adaptation.
            # ------------------------------------------------

            if bool(
                adaptation_result.diagnostics.get(
                    "reset_triggered",
                    False,
                )
            ):

                record_reset = getattr(
                    method,
                    "record_reset_location",
                    None,
                )

                if callable(record_reset):

                    record_reset(
                        year=int(year),
                        sample_index=int(index),
                        global_sample_index=int(
                            global_sample_offset
                            + index
                        ),
                        diagnostics=(
                            adaptation_result.diagnostics
                        ),
                    )

            logits = method.predict_logits(
                x
            )

            predictions.append(
                _prediction_from_single_logits(
                    logits
                )
            )

        elif (
            method.evaluation_order
            is EvaluationOrder.PREDICT_THEN_ADAPT
        ):

            logits = method.predict_logits(
                x
            )

            predictions.append(
                _prediction_from_single_logits(
                    logits
                )
            )

            adaptation_result = method.observe(
                x=x,
                y_aux=auxiliary_label,
            )

            # Keep the evaluator generic. A future
            # PREDICT_THEN_ADAPT method may also expose
            # reset diagnostics.
            if bool(
                adaptation_result.diagnostics.get(
                    "reset_triggered",
                    False,
                )
            ):

                record_reset = getattr(
                    method,
                    "record_reset_location",
                    None,
                )

                if callable(record_reset):

                    record_reset(
                        year=int(year),
                        sample_index=int(index),
                        global_sample_index=int(
                            global_sample_offset
                            + index
                        ),
                        diagnostics=(
                            adaptation_result.diagnostics
                        ),
                    )

        else:

            raise RuntimeError(
                "Unsupported evaluation order: "
                f"{method.evaluation_order!r}"
            )

    return np.asarray(
        predictions,
        dtype=np.int64,
    )

def _build_batch_ranges(
    n_samples: int,
    batch_size: int,
    *,
    minimum_batch_size: int,
) -> list[tuple[int, int]]:
    """
    Build contiguous temporal batches without reordering.

    If the final remainder is smaller than the required
    minimum batch size, it is merged into the previous batch.
    """

    if n_samples < minimum_batch_size:
        raise ValueError(
            "Stream contains fewer samples than "
            f"minimum_batch_size={minimum_batch_size}."
        )

    if batch_size < minimum_batch_size:
        raise ValueError(
            "Configured batch_size must be >= "
            f"{minimum_batch_size}."
        )

    ranges: list[
        tuple[int, int]
    ] = []

    start = 0

    while start < n_samples:

        end = min(
            start + batch_size,
            n_samples,
        )

        remainder = (
            n_samples
            - end
        )

        if (
            0
            < remainder
            < minimum_batch_size
        ):
            end = n_samples

        ranges.append(
            (
                start,
                end,
            )
        )

        start = end

    return ranges


def _batchwise_predictions(
    method: TTAMethod,
    X: np.ndarray,
    y_aux: np.ndarray | None,
    batch_size: int,
    minimum_batch_size: int,
) -> np.ndarray:
    """
    Generic contiguous batch-wise evaluator.

    This is currently required by TENT because target-batch
    BatchNorm statistics need batches of at least two samples.

    Ordering still comes exclusively from EvaluationOrder.
    """

    predictions: list[
        np.ndarray
    ] = []

    ranges = _build_batch_ranges(
        n_samples=len(X),
        batch_size=batch_size,
        minimum_batch_size=(
            minimum_batch_size
        ),
    )

    for start, end in ranges:

        x_batch = X[
            start:end
        ]

        auxiliary_batch = (
            None
            if y_aux is None
            else y_aux[
                start:end
            ]
        )

        if (
            method.evaluation_order
            is EvaluationOrder.ADAPT_THEN_PREDICT
        ):

            method.observe(
                x=x_batch,
                y_aux=auxiliary_batch,
            )

            logits = method.predict_logits(
                x_batch
            )

        elif (
            method.evaluation_order
            is EvaluationOrder.PREDICT_THEN_ADAPT
        ):

            logits = method.predict_logits(
                x_batch
            )

            method.observe(
                x=x_batch,
                y_aux=auxiliary_batch,
            )

        else:

            raise RuntimeError(
                "Unsupported evaluation order: "
                f"{method.evaluation_order!r}"
            )

        batch_predictions = (
            logits.argmax(
                dim=1
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )

        predictions.append(
            batch_predictions
        )

    return np.concatenate(
        predictions,
        axis=0,
    )


def _stream_batch_size(
    method: TTAMethod,
) -> int:

    return int(
        getattr(
            method,
            "stream_batch_size",
            1,
        )
    )


def _minimum_stream_batch_size(
    method: TTAMethod,
) -> int:

    return int(
        getattr(
            method,
            "minimum_stream_batch_size",
            1,
        )
    )


def evaluate_tta_year(
    method: TTAMethod,
    X: np.ndarray,
    y_main: np.ndarray,
    y_aux: np.ndarray | None,
    year: int,
    global_sample_offset: int = 0,
) -> EvaluationRecord:
    """
    Evaluate one chronological year.

    y_main is never passed to the TTA method. It is accessed
    only after every prediction has been produced and is used
    solely for metric computation.

    global_sample_offset is retained in the public signature
    because result/logging layers may use it later. Adaptation
    itself does not need it.
    """

    X = _as_numpy(
        X
    ).astype(
        np.float32,
        copy=False,
    )

    y_main = _as_numpy(
        y_main
    ).astype(
        np.int64,
        copy=False,
    )

    if y_aux is not None:
        y_aux = _as_numpy(
            y_aux
        ).astype(
            np.int64,
            copy=False,
        )

    if len(X) != len(y_main):
        raise ValueError(
            "X and y_main must have equal length."
        )

    if (
        y_aux is not None
        and len(X) != len(y_aux)
    ):
        raise ValueError(
            "X and y_aux must have equal length."
        )

    if len(X) == 0:
        raise ValueError(
            "Cannot evaluate an empty year."
        )

    if (
        method.requires_aux_labels
        and y_aux is None
    ):
        raise ValueError(
            f"{method.method_name} requires "
            "auxiliary labels."
        )

    batch_size = _stream_batch_size(
        method
    )

    minimum_batch_size = (
        _minimum_stream_batch_size(
            method
        )
    )

    if batch_size < 1:
        raise RuntimeError(
            f"{method.method_name} declared invalid "
            f"stream_batch_size={batch_size}."
        )

    if minimum_batch_size < 1:
        raise RuntimeError(
            f"{method.method_name} declared invalid "
            "minimum_stream_batch_size="
            f"{minimum_batch_size}."
        )

    if batch_size == 1:

        if minimum_batch_size != 1:
            raise RuntimeError(
                f"{method.method_name} requests "
                "sample-wise evaluation but declares "
                "minimum_stream_batch_size="
                f"{minimum_batch_size}."
            )

        predictions = (
            _samplewise_predictions(
                method=method,
                X=X,
                y_aux=y_aux,
                year=int(year),
                global_sample_offset=int(
                    global_sample_offset
                ),
            )
        )

    else:

        predictions = (
            _batchwise_predictions(
                method=method,
                X=X,
                y_aux=y_aux,
                batch_size=batch_size,
                minimum_batch_size=(
                    minimum_batch_size
                ),
            )
        )

    if len(predictions) != len(
        y_main
    ):
        raise RuntimeError(
            "Evaluator produced an incorrect "
            "number of predictions."
        )

    return EvaluationRecord(
        year=int(year),
        method=method.method_name,
        accuracy=accuracy(
            y_main,
            predictions,
        ),
        n_samples=int(
            len(y_main)
        ),
    )


def evaluate_tta_stream(
    method: TTAMethod,
    years: list[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray | None,
        ]
    ],
) -> EvaluationResult:
    """
    Evaluate a chronological multi-year stream.

    Reset semantics belong to the TTA method itself.

    The evaluator calls on_year_start() exactly once for each
    year and never performs an implicit annual reset.
    """

    records: list[
        EvaluationRecord
    ] = []

    global_sample_offset = 0

    for (
        year,
        X,
        y_main,
        y_aux,
    ) in years:

        method.on_year_start(
            int(year)
        )

        record = evaluate_tta_year(
            method=method,
            X=X,
            y_main=y_main,
            y_aux=y_aux,
            year=int(year),
            global_sample_offset=(
                global_sample_offset
            ),
        )

        global_sample_offset += int(
            len(X)
        )

        records.append(
            record
        )

    return EvaluationResult(
        records=tuple(
            records
        ),
        diagnostics=(
            method.diagnostics()
        ),
    )