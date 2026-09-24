from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

from metadata_tta.evaluation import (
    EvaluationRecord,
    EvaluationResult,
)


class ResultsWriter:
    """
    Persistence layer for one V3 experiment run.

    Expected layout:

        run/
          run_config.yaml
          manifest.json
          run.log

          tuning/

          checkpoints/
            source_model.pt
            aux_head.pt

          results/
            id/
              frozen/
                yearly_results.csv
                summary.json

            tta/
              frozen/
              metadata_episodic/
              ...

            supervised_reference/
              yearly_results.csv
              summary.json

            summary.csv
    """

    def __init__(
        self,
        run_directory: str | Path,
    ) -> None:

        self.run_directory = Path(
            run_directory
        )

        self.tuning_directory = (
            self.run_directory
            / "tuning"
        )

        self.checkpoints_directory = (
            self.run_directory
            / "checkpoints"
        )

        self.results_directory = (
            self.run_directory
            / "results"
        )

        self.id_directory = (
            self.results_directory
            / "id"
        )

        self.tta_directory = (
            self.results_directory
            / "tta"
        )

        self.supervised_reference_directory = (
            self.results_directory
            / "supervised_reference"
        )

        for directory in (
            self.run_directory,
            self.tuning_directory,
            self.checkpoints_directory,
            self.results_directory,
            self.id_directory,
            self.tta_directory,
            self.supervised_reference_directory,
        ):
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    def write_config(
        self,
        config: dict[str, Any],
    ) -> Path:

        path = (
            self.run_directory
            / "run_config.yaml"
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:

            yaml.safe_dump(
                config,
                handle,
                sort_keys=False,
            )

        return path

    def write_manifest(
        self,
        manifest: dict[str, Any],
    ) -> Path:

        path = (
            self.run_directory
            / "manifest.json"
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                manifest,
                handle,
                indent=2,
                sort_keys=True,
                default=str,
            )

        return path

    @staticmethod
    def _write_records_csv(
        path: Path,
        records: Iterable[
            EvaluationRecord
        ],
    ) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "year",
                    "method",
                    "accuracy",
                    "n_samples",
                ],
            )

            writer.writeheader()

            for record in records:
                writer.writerow(
                    asdict(
                        record
                    )
                )

    @staticmethod
    def _summary_for_records(
        records: Iterable[
            EvaluationRecord
        ],
    ) -> dict[str, Any]:

        records = tuple(
            records
        )

        if not records:
            return {
                "n_years": 0,
                "mean_accuracy": None,
                "std_accuracy": None,
                "worst_accuracy": None,
                "best_accuracy": None,
            }

        values = [
            float(record.accuracy)
            for record in records
        ]

        import numpy as np

        array = np.asarray(
            values,
            dtype=np.float64,
        )

        weights = np.asarray(
            [
                int(record.n_samples)
                for record in records
            ],
            dtype=np.float64,
        )

        if np.sum(weights) <= 0:
            raise RuntimeError(
                "Cannot compute sample-weighted accuracy "
                "with zero total samples."
            )

        return {
            "n_years": len(records),
            "mean_accuracy": float(
                np.average(
                    array,
                    weights=weights,
                )
            ),
            "std_accuracy": float(
                np.std(array)
            ),
            "worst_accuracy": float(
                np.min(array)
            ),
            "best_accuracy": float(
                np.max(array)
            ),
        }

    @staticmethod
    def _write_json(
        path: Path,
        payload: Any,
    ) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                default=str,
            )

    def write_id_frozen(
        self,
        records: Iterable[
            EvaluationRecord
        ],
    ) -> None:

        records = tuple(
            records
        )

        directory = (
            self.id_directory
            / "frozen"
        )

        self._write_records_csv(
            directory
            / "yearly_results.csv",
            records,
        )

        self._write_json(
            directory
            / "summary.json",
            self._summary_for_records(
                records
            ),
        )

    def write_frozen_ood(
        self,
        records: Iterable[
            EvaluationRecord
        ],
    ) -> None:

        records = tuple(
            records
        )

        directory = (
            self.tta_directory
            / "frozen"
        )

        self._write_records_csv(
            directory
            / "yearly_results.csv",
            records,
        )

        self._write_json(
            directory
            / "summary.json",
            self._summary_for_records(
                records
            ),
        )

    def write_tta_result(
        self,
        method_name: str,
        result: EvaluationResult,
    ) -> None:

        directory = (
            self.tta_directory
            / method_name
        )

        self._write_records_csv(
            directory
            / "yearly_results.csv",
            result.records,
        )

        summary = (
            self._summary_for_records(
                result.records
            )
        )

        summary[
            "method"
        ] = method_name

        self._write_json(
            directory
            / "summary.json",
            summary,
        )

        self._write_json(
            directory
            / "diagnostics.json",
            result.diagnostics,
        )

    def write_supervised_reference(
        self,
        records: Iterable[
            EvaluationRecord
        ],
    ) -> None:

        records = tuple(
            records
        )

        self._write_records_csv(
            self.supervised_reference_directory
            / "yearly_results.csv",
            records,
        )

        summary = (
            self._summary_for_records(
                records
            )
        )

        summary[
            "method"
        ] = "supervised_reference"

        self._write_json(
            self.supervised_reference_directory
            / "summary.json",
            summary,
        )

    def write_global_summary(
        self,
        rows: list[
            dict[str, Any]
        ],
    ) -> Path:

        path = (
            self.results_directory
            / "summary.csv"
        )

        fieldnames = [
            "method",
            "family",
            "n_years",
            "mean_accuracy",
            "std_accuracy",
            "worst_accuracy",
            "best_accuracy",
            "delta_vs_frozen",
        ]

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for row in rows:
                writer.writerow(
                    {
                        key: row.get(
                            key,
                            "",
                        )
                        for key
                        in fieldnames
                    }
                )

        return path

    def save_source_model(
        self,
        model: torch.nn.Module,
    ) -> Path:

        path = (
            self.checkpoints_directory
            / "source_model.pt"
        )

        torch.save(
            model.state_dict(),
            path,
        )

        return path

    def save_aux_head(
        self,
        double_model: torch.nn.Module,
    ) -> Path:
        """
        Store only the trained Aux Head parameters.

        The full source main path is already represented by
        source_model.pt.
        """

        if not hasattr(
            double_model,
            "aux_head",
        ):
            raise RuntimeError(
                "Double model does not expose aux_head."
            )

        path = (
            self.checkpoints_directory
            / "aux_head.pt"
        )

        torch.save(
            double_model.aux_head.state_dict(),
            path,
        )

        return path