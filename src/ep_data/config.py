"""Dataset registry loading and validation."""

from __future__ import annotations

import json
import keyword
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ep_data.exceptions import ConfigurationError, UnknownDatasetError

DOI_PATTERN = re.compile(r"^10\.17182/hepdata\.\d+\.v[1-9]\d*$")
DATASET_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    doi: str
    parser: str | None
    diagnostics: str | None


def _validate_module_name(key: str, field: str, value: Any) -> str | None:
    if value is not None and (
        not isinstance(value, str)
        or not value.isidentifier()
        or keyword.iskeyword(value)
    ):
        raise ConfigurationError(
            f"Dataset {key!r} has an invalid {field} module name: {value!r}"
        )
    return value


def _validate_entry(key: str, value: Any) -> DatasetConfig:
    if not isinstance(key, str) or not DATASET_KEY_PATTERN.fullmatch(key):
        raise ConfigurationError(f"Unsafe or invalid dataset key: {key!r}")
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration for {key!r} must be an object")
    if "script" in value:
        raise ConfigurationError(
            f"Dataset {key!r} uses unsupported legacy field 'script'; use 'parser'"
        )
    if "doi" not in value:
        raise ConfigurationError(f"Dataset {key!r} is missing required field 'doi'")
    if "parser" not in value:
        raise ConfigurationError(f"Dataset {key!r} is missing required field 'parser'")
    if "diagnostics" not in value:
        raise ConfigurationError(
            f"Dataset {key!r} is missing required field 'diagnostics'"
        )

    doi = value["doi"]
    if not isinstance(doi, str) or not DOI_PATTERN.fullmatch(doi):
        raise ConfigurationError(f"Dataset {key!r} has an invalid HEPData DOI: {doi!r}")

    parser = _validate_module_name(key, "parser", value["parser"])
    diagnostics = _validate_module_name(key, "diagnostics", value["diagnostics"])
    return DatasetConfig(doi=doi, parser=parser, diagnostics=diagnostics)


def load_config(path: Path) -> dict[str, DatasetConfig]:
    """Load and validate a JSON dataset registry without importing parsers."""
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as stream:
            raw = json.load(stream)
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration file does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Could not load configuration {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration top level must be an object")
    return {key: _validate_entry(key, value) for key, value in raw.items()}


def get_dataset(config: dict[str, DatasetConfig], key: str) -> DatasetConfig:
    """Return a configured dataset or raise an informative error."""
    try:
        return config[key]
    except KeyError as exc:
        available = ", ".join(config) or "<none>"
        raise UnknownDatasetError(
            f"Unknown dataset {key!r}; configured datasets: {available}"
        ) from exc
