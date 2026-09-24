from __future__ import annotations

import torch
import torch.nn as nn

from .adapter import ResidualBottleneckAdapter


class SingleHeadClassifier(nn.Module):
    """
    Single-head classifier.

    With shared trunk:

        embedding
            -> residual adapter
            -> Linear
            -> BatchNorm
            -> ReLU
            -> Dropout
            -> main head

    Without shared trunk:

        embedding
            -> residual adapter
            -> BatchNorm
            -> main head

    BatchNorm is intentionally kept also without the trunk
    because TENT adapts its affine parameters.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes_main: int,
        shared_hidden_dim: int,
        bottleneck_dim: int,
        dropout: float,
        use_shared_trunk: bool,
    ) -> None:
        super().__init__()

        self.use_shared_trunk = bool(
            use_shared_trunk
        )

        self.online_adapter = (
            ResidualBottleneckAdapter(
                input_dim=input_dim,
                bottleneck_dim=bottleneck_dim,
            )
        )

        if self.use_shared_trunk:

            self.feature_block = nn.Sequential(
                nn.Linear(
                    input_dim,
                    shared_hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm1d(
                    shared_hidden_dim
                ),
                nn.ReLU(),
                nn.Dropout(
                    dropout
                ),
            )

            head_input_dim = (
                shared_hidden_dim
            )

        else:

            self.feature_block = (
                nn.BatchNorm1d(
                    input_dim
                )
            )

            head_input_dim = (
                input_dim
            )

        self.main_head = nn.Linear(
            head_input_dim,
            n_classes_main,
        )

    def extract_shared_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        adapted = (
            self.online_adapter(x)
        )

        return self.feature_block(
            adapted
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        shared = (
            self.extract_shared_features(x)
        )

        return self.main_head(
            shared
        )