from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(
    seed: int,
    deterministic: bool = False,
) -> None:
    """
    Seed Python, NumPy and PyTorch.

    Parameters
    ----------
    seed:
        Global random seed.

    deterministic:
        If True, request deterministic PyTorch algorithms
        whenever possible.

        The default is False because strict deterministic
        mode can reduce performance and some operations may
        not support it on every backend.
    """

    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(
        seed
    )

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    if deterministic:
        torch.use_deterministic_algorithms(
            True
        )