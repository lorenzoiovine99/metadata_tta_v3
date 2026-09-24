from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


class CacheManager:
    """
    Cache used only for acceleration.

    Scientific results must never depend on the existence
    of cached files.
    """

    def __init__(
        self,
        root: str | Path,
        enabled: bool = True,
        version: str = "v1",
    ) -> None:
        self.root = Path(root)
        self.enabled = bool(enabled)
        self.version = str(version)

        if self.enabled:
            self.root.mkdir(
                parents=True,
                exist_ok=True,
            )

    def _key(
        self,
        namespace: str,
        payload: dict[str, Any],
    ) -> str:

        serialized = json.dumps(
            {
                "version": self.version,
                "namespace": namespace,
                "payload": payload,
            },
            sort_keys=True,
            default=str,
        )

        digest = hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()[:16]

        return digest

    def path(
        self,
        namespace: str,
        payload: dict[str, Any],
    ) -> Path:

        key = self._key(
            namespace,
            payload,
        )

        directory = (
            self.root
            / namespace
        )

        return directory / f"{key}.pt"

    def load(
        self,
        namespace: str,
        payload: dict[str, Any],
    ) -> Any | None:

        if not self.enabled:
            return None

        path = self.path(
            namespace,
            payload,
        )

        if not path.exists():
            return None

        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    def save(
        self,
        namespace: str,
        payload: dict[str, Any],
        value: Any,
    ) -> Path | None:

        if not self.enabled:
            return None

        path = self.path(
            namespace,
            payload,
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            value,
            path,
        )

        return path