from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Invalid V3 experiment configuration."""


def _load_yaml_mapping(
    path: str | Path,
) -> dict[str, Any]:

    path = Path(path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = yaml.safe_load(handle)

    if data is None:
        raise ConfigError(
            f"Configuration file is empty: {path}"
        )

    if not isinstance(data, dict):
        raise ConfigError(
            f"Configuration root must be a mapping: {path}"
        )

    return data


def _deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Recursive dictionary merge.

    Values in `override` win.
    """

    result = copy.deepcopy(
        dict(base)
    )

    for key, value in override.items():

        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(
                result[key],
                value,
            )

        else:
            result[key] = copy.deepcopy(
                value
            )

    return result


class ExperimentConfig:
    """
    Canonical resolved V3 configuration.

    Expected top-level sections:

        dataset
        protocol
        experiment
        output
        logging
        model
        training
        methods
        supervised_reference
        checks

    Dataset YAML and experiment YAML are composed before this
    object is constructed.
    """

    VALID_MODES = {
        "tune",
        "test",
        "tune_and_test",
    }

    def __init__(
        self,
        data: Mapping[str, Any],
    ) -> None:

        if not isinstance(
            data,
            Mapping,
        ):
            raise ConfigError(
                "ExperimentConfig requires a mapping."
            )

        self._data = copy.deepcopy(
            dict(data)
        )

        self._validate()

    # ========================================================
    # VALIDATION
    # ========================================================

    def _require_section(
        self,
        name: str,
    ) -> dict[str, Any]:

        value = self._data.get(
            name
        )

        if not isinstance(
            value,
            dict,
        ):
            raise ConfigError(
                f"Missing or invalid section: {name}"
            )

        return value

    def _validate(self) -> None:

        required_sections = (
            "dataset",
            "protocol",
            "experiment",
            "output",
            "logging",
            "model",
            "training",
            "methods",
            "supervised_reference",
            "checks",
        )

        for name in required_sections:
            self._require_section(
                name
            )

        dataset = self.section(
            "dataset"
        )

        for key in (
            "name",
            "data_root",
            "n_classes_main",
            "n_classes_aux",
            "split",
        ):
            if key not in dataset:
                raise ConfigError(
                    f"dataset.{key} is required."
                )

        split = dataset[
            "split"
        ]

        if not isinstance(
            split,
            dict,
        ):
            raise ConfigError(
                "dataset.split must be a mapping."
            )

        for key in (
            "test_size",
            "validation_size_within_train",
        ):
            if key not in split:
                raise ConfigError(
                    f"dataset.split.{key} is required."
                )

        protocol = self.section(
            "protocol"
        )

        for phase in (
            "pseudo_source",
            "pseudo_ood",
            "final_source",
            "real_ood",
        ):
            phase_config = protocol.get(
                phase
            )

            if not isinstance(
                phase_config,
                dict,
            ):
                raise ConfigError(
                    f"protocol.{phase} must be a mapping."
                )

            for key in (
                "start_year",
                "end_year",
            ):
                if key not in phase_config:
                    raise ConfigError(
                        f"protocol.{phase}.{key} is required."
                    )

            if int(
                phase_config["start_year"]
            ) > int(
                phase_config["end_year"]
            ):
                raise ConfigError(
                    f"protocol.{phase}: start_year "
                    "must be <= end_year."
                )

        experiment = self.section(
            "experiment"
        )

        for key in (
            "mode",
            "experiment_seed",
            "split_seed",
        ):
            if key not in experiment:
                raise ConfigError(
                    f"experiment.{key} is required."
                )

        mode = str(
            experiment["mode"]
        ).strip().lower()

        if mode not in self.VALID_MODES:
            raise ConfigError(
                "experiment.mode must be one of "
                f"{sorted(self.VALID_MODES)}; "
                f"got {mode!r}."
            )

        model = self.section(
            "model"
        )

        for key in (
            "shared_hidden_dim",
            "adapter_bottleneck_dim",
            "dropout",
            "use_shared_trunk",
        ):
            if key not in model:
                raise ConfigError(
                    f"model.{key} is required."
                )

        training = self.section(
            "training"
        )

        for key in (
            "base",
            "single_head",
            "double_head",
            "temporal",
        ):
            if not isinstance(
                training.get(key),
                dict,
            ):
                raise ConfigError(
                    f"training.{key} must be a mapping."
                )

        base = training[
            "base"
        ]

        for key in (
            "learning_rate",
            "weight_decay",
            "epochs",
            "batch_size",
        ):
            if key not in base:
                raise ConfigError(
                    f"training.base.{key} is required."
                )

        methods = self.section(
            "methods"
        )

        for method_name, method_config in (
            methods.items()
        ):
            if not isinstance(
                method_config,
                dict,
            ):
                raise ConfigError(
                    f"methods.{method_name} "
                    "must be a mapping."
                )

            if "enabled" not in method_config:
                raise ConfigError(
                    f"methods.{method_name}.enabled "
                    "is required."
                )

        supervised_reference = self.section(
            "supervised_reference"
        )

        if (
            supervised_reference.get(
                "initialization"
            )
            != "final_source_model"
        ):
            raise ConfigError(
                "V3 supervised_reference.initialization "
                "must be 'final_source_model'."
            )

        if not bool(
            supervised_reference.get(
                "independent_per_year",
                False,
            )
        ):
            raise ConfigError(
                "V3 supervised reference must use "
                "independent_per_year=true."
            )

    # ========================================================
    # GENERIC ACCESS
    # ========================================================

    def as_dict(
        self,
    ) -> dict[str, Any]:

        return copy.deepcopy(
            self._data
        )

    def section(
        self,
        name: str,
    ) -> dict[str, Any]:

        value = self._data.get(
            name
        )

        if not isinstance(
            value,
            dict,
        ):
            raise KeyError(
                f"Missing configuration section {name!r}."
            )

        # Return a copy deliberately.
        #
        # Existing model/training helpers sometimes merge
        # head-specific overrides into this dictionary.
        return copy.deepcopy(
            value
        )

    def get(
        self,
        *keys: str,
        default: Any = None,
    ) -> Any:
        """
        Nested lookup.

        Examples:

            config.get("dataset", "split")
            config.get("model", "single_head", "dropout")
        """

        if not keys:
            return self.as_dict()

        current: Any = self._data

        for key in keys:

            if not isinstance(
                current,
                Mapping,
            ):
                return copy.deepcopy(
                    default
                )

            if key not in current:
                return copy.deepcopy(
                    default
                )

            current = current[
                key
            ]

        return copy.deepcopy(
            current
        )

    def with_overrides(
        self,
        overrides: Mapping[str, Any],
    ) -> "ExperimentConfig":
        """
        Apply dot-separated paths and return a NEW config.

        Example:

            {
                "training.base.learning_rate": 3e-4,
                "methods.tent.batch_size": 64,
            }
        """

        updated = self.as_dict()

        for path, value in (
            overrides.items()
        ):
            components = str(
                path
            ).split(".")

            if not components or any(
                not component
                for component in components
            ):
                raise ConfigError(
                    f"Invalid override path: {path!r}"
                )

            current = updated

            for component in (
                components[:-1]
            ):

                existing = current.get(
                    component
                )

                if existing is None:
                    current[
                        component
                    ] = {}

                elif not isinstance(
                    existing,
                    dict,
                ):
                    raise ConfigError(
                        "Cannot descend through "
                        f"non-mapping override path "
                        f"{path!r} at {component!r}."
                    )

                current = current[
                    component
                ]

            current[
                components[-1]
            ] = copy.deepcopy(
                value
            )

        return ExperimentConfig(
            updated
        )

    # ========================================================
    # V3 PROPERTIES
    # ========================================================

    @property
    def dataset_name(
        self,
    ) -> str:

        return str(
            self._data[
                "dataset"
            ][
                "name"
            ]
        )

    @property
    def data_root(
        self,
    ) -> Path:

        return Path(
            self._data[
                "dataset"
            ][
                "data_root"
            ]
        ).expanduser()

    @property
    def cache_root(
        self,
    ) -> Path:

        raw = self._data[
            "dataset"
        ].get(
            "cache_root",
            ".cache",
        )

        return Path(
            raw
        ).expanduser()

    @property
    def results_root(
        self,
    ) -> Path:

        return Path(
            self._data[
                "output"
            ].get(
                "root",
                "results",
            )
        ).expanduser()

    @property
    def mode(
        self,
    ) -> str:

        return str(
            self._data[
                "experiment"
            ][
                "mode"
            ]
        ).strip().lower()

    @property
    def experiment_seed(
        self,
    ) -> int:

        return int(
            self._data[
                "experiment"
            ][
                "experiment_seed"
            ]
        )

    @property
    def split_seed(
        self,
    ) -> int:

        return int(
            self._data[
                "experiment"
            ][
                "split_seed"
            ]
        )

    # Compatibility alias used by generic supervised helpers.
    #
    # In V3 "seed" always means experiment_seed.
    @property
    def seed(
        self,
    ) -> int:

        return self.experiment_seed

    @property
    def enabled_methods(
        self,
    ) -> list[str]:

        result = []

        for name, method_config in (
            self._data[
                "methods"
            ].items()
        ):
            if bool(
                method_config.get(
                    "enabled",
                    False,
                )
            ):
                result.append(
                    str(name)
                )

        return result

    def method_enabled(
        self,
        name: str,
    ) -> bool:

        method_config = (
            self._data[
                "methods"
            ].get(
                name
            )
        )

        return bool(
            isinstance(
                method_config,
                Mapping,
            )
            and method_config.get(
                "enabled",
                False,
            )
        )

    def method_config(
        self,
        name: str,
    ) -> dict[str, Any]:

        method_config = (
            self._data[
                "methods"
            ].get(
                name
            )
        )

        if not isinstance(
            method_config,
            dict,
        ):
            raise KeyError(
                f"Unknown method {name!r}."
            )

        return copy.deepcopy(
            method_config
        )

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(
        self,
    ) -> str:

        return (
            "ExperimentConfig("
            f"dataset={self.dataset_name!r}, "
            f"mode={self.mode!r}, "
            f"experiment_seed={self.experiment_seed}, "
            f"split_seed={self.split_seed}"
            ")"
        )


def compose_config(
    *,
    dataset_path: str | Path,
    experiment_path: str | Path,
    overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    """
    Compose:

        dataset YAML
        +
        experiment YAML
        +
        optional runtime overrides

    Dataset YAML contains both:
        dataset:
        protocol:

    Experiment YAML contains model/training/method/run settings.
    """

    dataset_config = _load_yaml_mapping(
        dataset_path
    )

    experiment_config = _load_yaml_mapping(
        experiment_path
    )

    merged = _deep_merge(
        experiment_config,
        dataset_config,
    )

    config = ExperimentConfig(
        merged
    )

    if overrides:
        config = config.with_overrides(
            overrides
        )

    return config


def load_config(
    path: str | Path,
) -> ExperimentConfig:
    """
    Load an already-resolved V3 configuration.

    For normal runs prefer compose_config().
    """

    return ExperimentConfig(
        _load_yaml_mapping(
            path
        )
    )


def load_yaml(
    path: str | Path,
) -> dict[str, Any]:
    """
    Public generic YAML loader for tuning/batch configuration.
    """

    return _load_yaml_mapping(
        path
    )


__all__ = [
    "ConfigError",
    "ExperimentConfig",
    "compose_config",
    "load_config",
    "load_yaml",
]