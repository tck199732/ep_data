"""Parser for HEPData record ins110317 (Harvard/Cornell/CEA, 1977)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ep_data.models import (
    Kinematics,
    Measurement,
    Metadata,
    Qualifier,
    Xsect,
)
from ep_data.utils import write_json_atomic

DATASET_KEY = "harvard_ep_1977"
DOI = "10.17182/hepdata.24604.v1"
RECORD_ID = "110317"
_REACTION_OUTPUT_NAMES = {
    "GAMMA* P --> K+ LAMBDA": "lambda",
    "GAMMA* P --> K+ SIGMA0": "sigma0",
    "GAMMA* N --> K+ SIGMA-": "sigma_minus",
}

_TABLE2_REACTIONS = {
    "D(SIG(Q=LAM))/DOMEGA": "GAMMA* P --> K+ LAMBDA",
    "D(SIG(Q=SIG))/DOMEGA": "GAMMA* P --> K+ SIGMA0",
}
_TABLE3_REACTIONS = {
    "D(SIG(Q=SIG0))/DOMEGA": "GAMMA* P --> K+ SIGMA0",
    "D(SIG(Q=SIG-))/DOMEGA": "GAMMA* N --> K+ SIGMA-",
}


@dataclass(frozen=True, slots=True)
class _ExperimentBlock:
    start: int
    stop: int
    experiment: str
    year: int
    reference: str


_TABLE2_BLOCKS = (
    _ExperimentBlock(
        0,
        11,
        "Cornell",
        1977,
        "Phys. Rev. D 15, 594 (1977); DOI:10.1103/PhysRevD.15.594",
    ),
    _ExperimentBlock(
        11,
        15,
        "Harvard-Cornell",
        1974,
        "Bebek et al., Phys. Rev. Lett. 32, 21 (1974)",
    ),
    _ExperimentBlock(
        15,
        23,
        "CEA",
        1972,
        "Brown et al., Phys. Rev. Lett. 28, 1086 (1972)",
    ),
)
_TABLE3_BLOCK = _ExperimentBlock(
    0,
    6,
    "Cornell",
    1977,
    "Phys. Rev. D 15, 594 (1977); DOI:10.1103/PhysRevD.15.594",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a YAML mapping")
    return value


def _load_submission(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            documents = list(yaml.safe_load_all(stream))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read {path.name}: {exc}") from exc
    if not documents or not isinstance(documents[0], dict):
        raise ValueError(f"{path.name} must begin with a YAML mapping")
    return documents[0]


def _variables_by_name(
    table: dict[str, Any], section: str, *, expected_length: int
) -> dict[str, dict[str, Any]]:
    variables = table.get(section)
    if not isinstance(variables, list):
        raise ValueError(f"{section} must be a list")

    result: dict[str, dict[str, Any]] = {}
    for variable in variables:
        if not isinstance(variable, dict):
            raise ValueError(f"Entries in {section} must be mappings")
        header = variable.get("header")
        values = variable.get("values")
        if not isinstance(header, dict) or not isinstance(header.get("name"), str):
            raise ValueError(f"An entry in {section} has an invalid header")
        name = header["name"]
        if name in result:
            raise ValueError(f"Duplicate variable {name!r} in {section}")
        if not isinstance(values, list) or len(values) != expected_length:
            count = len(values) if isinstance(values, list) else 0
            raise ValueError(
                f"{name!r} contains {count} values; expected {expected_length}"
            )
        result[name] = variable
    return result


def _number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    return float(value)


def _kinematics(
    variables: dict[str, dict[str, Any]], block: _ExperimentBlock
) -> Kinematics:
    required = {"THETA", "W", "Q2", "EPS"}
    missing = required.difference(variables)
    if missing:
        raise ValueError(f"Missing kinematic variables: {', '.join(sorted(missing))}")

    selected = slice(block.start, block.stop)
    theta_entries = variables["THETA"]["values"][selected]
    theta_degrees = np.array(
        [
            (
                np.nan
                if entry.get("value") is None
                else _number(entry["value"], context="THETA value")
            )
            for entry in theta_entries
        ]
    )
    theta_center_degrees = np.array(
        [
            (
                _number(entry.get("low"), context="THETA low")
                + _number(entry.get("high"), context="THETA high")
            )
            / 2.0
            for entry in theta_entries
        ]
    )

    def values(name: str) -> np.ndarray:
        return np.array(
            [
                _number(entry.get("value"), context=f"{name} value")
                for entry in variables[name]["values"][selected]
            ]
        )

    return Kinematics(
        Q2=values("Q2"),
        W=values("W"),
        thetacm=np.deg2rad(theta_degrees),
        thetacm_center=np.deg2rad(theta_center_degrees),
        epsilon=values("EPS"),
    )


def _qualifiers(variable: dict[str, Any]) -> tuple[Qualifier, ...]:
    raw_qualifiers = variable.get("qualifiers", [])
    if not isinstance(raw_qualifiers, list):
        raise ValueError("qualifiers must be a list")
    qualifiers = []
    for raw in raw_qualifiers:
        if not isinstance(raw, dict) or "name" not in raw or "value" not in raw:
            raise ValueError("Each qualifier must contain name and value")
        qualifiers.append(
            Qualifier(name=str(raw["name"]), value=raw["value"], unit=raw.get("units"))
        )
    return tuple(qualifiers)


def _cross_section(variable: dict[str, Any], block: _ExperimentBlock) -> Xsect:
    central = []
    point_errors = []
    systematics = []
    lower_limits = []
    upper_limits = []

    for index, entry in enumerate(variable["values"][block.start : block.stop]):
        context = f"cross-section value at row {block.start + index + 1}"
        if not isinstance(entry, dict) or "value" not in entry:
            raise ValueError(f"Invalid {context}")
        raw_value = entry["value"]
        value = np.nan
        lower_limit = np.nan
        upper_limit = np.nan
        if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
            value = float(raw_value)
        elif isinstance(raw_value, str) and raw_value.startswith("<"):
            upper_limit = float(raw_value[1:])
        elif isinstance(raw_value, str) and raw_value.startswith(">"):
            lower_limit = float(raw_value[1:])
        elif raw_value != "-":
            raise ValueError(f"Unsupported {context}: {raw_value!r}")

        point_error = np.nan if np.isnan(value) else 0.0
        systematic = np.nan if np.isnan(value) else 0.0
        errors = entry.get("errors", [])
        if not isinstance(errors, list):
            raise ValueError(f"Errors for {context} must be a list")
        for error in errors:
            if not isinstance(error, dict) or "symerror" not in error:
                raise ValueError(f"Invalid error for {context}")
            label = error.get("label")
            raw_error = error["symerror"]
            if label is None:
                point_error = _number(raw_error, context=f"point error for {context}")
            elif str(label).startswith("sys,"):
                if not isinstance(raw_error, str) or not raw_error.endswith("%"):
                    raise ValueError(
                        f"Systematic error for {context} must be a percent"
                    )
                systematic = (
                    np.nan
                    if np.isnan(value)
                    else value * float(raw_error.removesuffix("%")) / 100.0
                )
            else:
                raise ValueError(f"Unexpected error label for {context}: {label!r}")

        central.append(value)
        point_errors.append(point_error)
        systematics.append(systematic)
        lower_limits.append(lower_limit)
        upper_limits.append(upper_limit)

    return Xsect(
        sigU=central,
        sigU_err=point_errors,
        sigU_sys_err_low=systematics,
        sigU_sys_err_high=systematics,
        sigU_lower_limit=lower_limits,
        sigU_upper_limit=upper_limits,
    )


def _measurements_from_table(
    table: dict[str, Any],
    *,
    table_id: str,
    expected_length: int,
    reactions: dict[str, str],
    blocks: tuple[_ExperimentBlock, ...],
) -> list[Measurement]:
    independent = _variables_by_name(
        table, "independent_variables", expected_length=expected_length
    )
    dependent = _variables_by_name(
        table, "dependent_variables", expected_length=expected_length
    )
    missing = set(reactions).difference(dependent)
    if missing:
        raise ValueError(f"Missing observables: {', '.join(sorted(missing))}")

    measurements = []
    for observable, reaction in reactions.items():
        variable = dependent[observable]
        header = variable["header"]
        if header.get("units") != "NB/SR":
            raise ValueError(
                f"Unexpected units for {observable!r}: {header.get('units')!r}"
            )
        for block in blocks:
            measurements.append(
                Measurement(
                    xsect=_cross_section(variable, block),
                    kinematics=_kinematics(independent, block),
                    metadata=Metadata(
                        reaction=reaction,
                        unit="nb/sr",
                        source_unit="NB/SR",
                        angle_unit="rad",
                        dataset_key=DATASET_KEY,
                        doi=DOI,
                        record_id=RECORD_ID,
                        table_id=table_id,
                        observable=observable,
                        experiment=block.experiment,
                        year=block.year,
                        reference=block.reference,
                        qualifiers=_qualifiers(variable),
                    ),
                )
            )
    return measurements


def _output_name(measurement: Measurement) -> str:
    table = measurement.metadata.table_id
    experiment = measurement.metadata.experiment
    reaction = measurement.metadata.reaction
    if table is None or experiment is None:
        raise ValueError("Measurement metadata must identify its table and experiment")
    try:
        reaction_name = _REACTION_OUTPUT_NAMES[reaction]
    except KeyError as exc:
        raise ValueError(
            f"No output name is defined for reaction {reaction!r}"
        ) from exc
    experiment_name = experiment.lower().replace("-", "_").replace(" ", "_")
    context = "_deuterium" if table == "Table 3" else ""
    return f"{experiment_name}{context}_{reaction_name}.json"


def parse(input_dir: Path, output_dir: Path) -> None:
    """Parse the approved unscaled cross sections into common JSON measurements."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    submission = _load_submission(input_dir / "submission.yaml")
    if submission.get("hepdata_doi") != DOI:
        raise ValueError(
            f"Expected HEPData DOI {DOI}, found {submission.get('hepdata_doi')!r}"
        )

    table1 = _load_yaml(input_dir / "Table1.yaml")
    if table1.get("dependent_variables") or table1.get("independent_variables"):
        raise ValueError("Table1.yaml was expected to be empty")

    table2 = _load_yaml(input_dir / "Table2.yaml")
    table3 = _load_yaml(input_dir / "Table3.yaml")
    measurements = _measurements_from_table(
        table2,
        table_id="Table 2",
        expected_length=23,
        reactions=_TABLE2_REACTIONS,
        blocks=_TABLE2_BLOCKS,
    )
    measurements.extend(
        _measurements_from_table(
            table3,
            table_id="Table 3",
            expected_length=6,
            reactions=_TABLE3_REACTIONS,
            blocks=(_TABLE3_BLOCK,),
        )
    )
    for measurement in measurements:
        write_json_atomic(output_dir / _output_name(measurement), measurement)
