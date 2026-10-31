"""Parser for the approved tables in HEPData record ins54849."""

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
from ep_data.parser.physics import (
    MASS_LAMBDA,
    MASS_SIGMA0,
    get_dt_dOmega,
)
from ep_data.utils import write_json_atomic

DATASET_KEY = "slac_yp_1969"
DOI = "10.17182/hepdata.3394.v1"
RECORD_ID = "54849"
SOURCE_OBSERVABLE = "D(SIG)/DT"
OUTPUT_OBSERVABLE = "D(SIG)/DOMEGA"
SOURCE_UNIT = "MB/GEV**2"
OUTPUT_UNIT = "nb/sr"
_MB_TO_NB = 1.0e6


@dataclass(frozen=True, slots=True)
class _TableSpec:
    number: int
    reaction: str
    reaction_name: str
    photon_momentum: float
    sqrt_s: float
    hyperon_mass: float
    points: int

    @property
    def output_name(self) -> str:
        momentum = f"{self.photon_momentum:g}".replace(".", "p")
        return f"{self.reaction_name}_{momentum}gev.json"


_TABLES = (
    _TableSpec(1, "GAMMA P --> K+ LAMBDA", "lambda", 5.0, 3.203, MASS_LAMBDA, 14),
    _TableSpec(2, "GAMMA P --> K+ LAMBDA", "lambda", 8.0, 3.986, MASS_LAMBDA, 17),
    _TableSpec(3, "GAMMA P --> K+ LAMBDA", "lambda", 11.0, 4.639, MASS_LAMBDA, 12),
    _TableSpec(4, "GAMMA P --> K+ LAMBDA", "lambda", 16.0, 5.558, MASS_LAMBDA, 13),
    _TableSpec(6, "GAMMA P --> K+ SIGMA0", "sigma0", 5.0, 3.203, MASS_SIGMA0, 13),
    _TableSpec(7, "GAMMA P --> K+ SIGMA0", "sigma0", 8.0, 3.986, MASS_SIGMA0, 14),
    _TableSpec(8, "GAMMA P --> K+ SIGMA0", "sigma0", 11.0, 4.639, MASS_SIGMA0, 10),
    _TableSpec(9, "GAMMA P --> K+ SIGMA0", "sigma0", 16.0, 5.558, MASS_SIGMA0, 11),
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


def _single_variable(
    table: dict[str, Any], section: str, *, expected_points: int
) -> dict[str, Any]:
    variables = table.get(section)
    if not isinstance(variables, list) or len(variables) != 1:
        raise ValueError(f"{section} must contain exactly one variable")
    variable = variables[0]
    if not isinstance(variable, dict):
        raise ValueError(f"The entry in {section} must be a mapping")
    values = variable.get("values")
    if not isinstance(values, list) or len(values) != expected_points:
        count = len(values) if isinstance(values, list) else 0
        raise ValueError(
            f"The variable in {section} contains {count} values; "
            f"expected {expected_points}"
        )
    return variable


def _qualifier_map(variable: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_qualifiers = variable.get("qualifiers")
    if not isinstance(raw_qualifiers, list):
        raise ValueError("Dependent-variable qualifiers must be a list")
    qualifiers = {}
    for raw in raw_qualifiers:
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
            raise ValueError("Each qualifier must have a string name")
        qualifiers[raw["name"]] = raw
    return qualifiers


def _number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    return float(value)


def _percent(value: Any, *, context: str) -> float:
    if not isinstance(value, str) or not value.endswith("%"):
        raise ValueError(f"{context} must be a percentage")
    return float(value.removesuffix("%")) / 100.0


def _parse_table(table: dict[str, Any], spec: _TableSpec) -> Measurement:
    independent = _single_variable(
        table, "independent_variables", expected_points=spec.points
    )
    dependent = _single_variable(
        table, "dependent_variables", expected_points=spec.points
    )
    independent_header = independent.get("header")
    dependent_header = dependent.get("header")
    if independent_header != {"name": "-T", "units": "GEV**2"}:
        raise ValueError(f"Table {spec.number} has an unexpected independent variable")
    if dependent_header != {"name": SOURCE_OBSERVABLE, "units": SOURCE_UNIT}:
        raise ValueError(f"Table {spec.number} has an unexpected dependent variable")

    qualifiers = _qualifier_map(dependent)
    expected_qualifiers = {
        "P": (str(spec.photon_momentum), "GeV"),
        "RE": (spec.reaction, None),
        "SQRT(S)": (str(spec.sqrt_s), "GeV"),
    }
    for name, (expected_value, expected_unit) in expected_qualifiers.items():
        raw = qualifiers.get(name)
        if raw is None or str(raw.get("value")) != expected_value:
            raise ValueError(f"Table {spec.number} has an unexpected {name} qualifier")
        if raw.get("units") != expected_unit:
            raise ValueError(f"Table {spec.number} has unexpected units for {name}")

    minus_t = np.array(
        [
            _number(item.get("value"), context=f"Table {spec.number} -T")
            for item in independent["values"]
        ]
    )
    source_values = []
    source_errors = []
    systematic_fractions = []
    for row, item in enumerate(dependent["values"], start=1):
        source_values.append(
            _number(item.get("value"), context=f"Table {spec.number} row {row}")
        )
        errors = item.get("errors")
        if not isinstance(errors, list) or len(errors) != 2:
            raise ValueError(f"Table {spec.number} row {row} must have two errors")
        point_error, systematic = errors
        if point_error.get("label") is not None or "symerror" not in point_error:
            raise ValueError(f"Table {spec.number} row {row} has invalid point error")
        if systematic.get("label") != "sys,BOYARSKI 69":
            raise ValueError(
                f"Table {spec.number} row {row} has unexpected systematic label"
            )
        source_errors.append(
            _number(
                point_error["symerror"],
                context=f"Table {spec.number} row {row} point error",
            )
        )
        systematic_fractions.append(
            _percent(
                systematic.get("symerror"),
                context=f"Table {spec.number} row {row} systematic error",
            )
        )

    q2 = np.zeros(spec.points)
    w = np.full(spec.points, spec.sqrt_s)
    conversion = _MB_TO_NB * get_dt_dOmega(q2, w, spec.hyperon_mass)
    cross_section = np.asarray(source_values) * conversion
    point_error = np.asarray(source_errors) * conversion
    systematic = cross_section * np.asarray(systematic_fractions)

    output_qualifiers = (
        Qualifier("P", spec.photon_momentum, "GeV"),
        Qualifier("RE", spec.reaction),
        Qualifier("SQRT(S)", spec.sqrt_s, "GeV"),
        Qualifier("OBSERVABLE", OUTPUT_OBSERVABLE, OUTPUT_UNIT),
    )
    return Measurement(
        xsect=Xsect(
            sigU=cross_section,
            sigU_err=point_error,
            sigU_sys_err_low=systematic,
            sigU_sys_err_high=systematic,
        ),
        kinematics=Kinematics(Q2=q2, W=w, t=minus_t),
        metadata=Metadata(
            reaction=spec.reaction,
            unit=OUTPUT_UNIT,
            source_unit=SOURCE_UNIT,
            dataset_key=DATASET_KEY,
            doi=DOI,
            record_id=RECORD_ID,
            table_id=f"Table {spec.number}",
            observable=OUTPUT_OBSERVABLE,
            experiment="SLAC",
            year=1969,
            reference=(
                "Boyarski et al., Phys. Rev. Lett. 22, 1131 (1969); "
                "DOI:10.1103/PhysRevLett.22.1131"
            ),
            qualifiers=output_qualifiers,
        ),
    )


def parse(input_dir: Path, output_dir: Path) -> None:
    """Convert approved Boyarski d-sigma/dt tables to d-sigma/d-Omega JSON."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    submission = _load_submission(input_dir / "submission.yaml")
    if submission.get("hepdata_doi") != DOI:
        raise ValueError(
            f"Expected HEPData DOI {DOI}, found {submission.get('hepdata_doi')!r}"
        )

    for spec in _TABLES:
        table = _load_yaml(input_dir / f"Table{spec.number}.yaml")
        measurement = _parse_table(table, spec)
        write_json_atomic(output_dir / spec.output_name, measurement)
