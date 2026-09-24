from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBottleneckAdapter(nn.Module):
    """
    Residual bottleneck adapter.

    A(x) = x + W_up(ReLU(W_down(x)))

    The up-projection is initialized to zero, therefore
    the adapter starts exactly as the identity function.

    This reproduces the architecture used in the original
    fMoW experiment.
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
    ) -> None:
        super().__init__()

        self.down = nn.Linear(
            input_dim,
            bottleneck_dim,
        )

        self.up = nn.Linear(
            bottleneck_dim,
            input_dim,
        )

        # Original initialization.
        nn.init.kaiming_uniform_(
            self.down.weight,
            nonlinearity="relu",
        )

        nn.init.zeros_(
            self.down.bias
        )

        # Zero up-projection => initial adapter is identity.
        nn.init.zeros_(
            self.up.weight
        )

        nn.init.zeros_(
            self.up.bias
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = self.down(
            x
        )

        residual = F.relu(
            residual
        )

        residual = self.up(
            residual
        )

        return x + residual