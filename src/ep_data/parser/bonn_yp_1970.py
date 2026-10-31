"""Parser for the Bonn photoproduction data in HEPData record ins63868."""

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
from ep_data.parser.physics import MASS_PROTON
from ep_data.utils import write_json_atomic

DATASET_KEY = "bonn_yp_1970"
DOI = "10.17182/hepdata.16698.v1"
RECORD_ID = "63868"
OBSERVABLE = "D(SIG)/DOMEGA"
SOURCE_UNIT = "MUB/SR"
OUTPUT_UNIT = "nb/sr"
_MUB_TO_NB = 1.0e3


@dataclass(frozen=True, slots=True)
class _TableSpec:
    number: int
    reaction: str
    points: int
    sqrt_s_qualifier: str
    output_name: str
    fixed_photon_energy: float | None = None


_TABLES = (
    _TableSpec(
        1,
        "GAMMA P --> K+ LAMBDA",
        11,
        "1.822",
        "lambda_1p3gev.json",
        1.3,
    ),
    _TableSpec(
        2,
        "GAMMA P --> K+ LAMBDA",
        7,
        "1.836-1.844",
        "lambda_1p327_1p344gev.json",
    ),
    _TableSpec(
        3,
        "GAMMA P --> K+ SIGMA0",
        9,
        "1.897",
        "sigma0_1p45gev.json",
        1.45,
    ),
    _TableSpec(
        4,
        "GAMMA P --> K+ SIGMA0",
        3,
        "1.877-1.88",
        "sigma0_1p410_1p416gev.json",
    ),
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
    table: dict[str, Any], section: str, *, expected_points: int
) -> dict[str, dict[str, Any]]:
    variables = table.get(section)
    if not isinstance(variables, list):
        raise ValueError(f"{section} must be a list")
    result = {}
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
        if not isinstance(values, list) or len(values) != expected_points:
            count = len(values) if isinstance(values, list) else 0
            raise ValueError(
                f"{name!r} contains {count} values; expected {expected_points}"
            )
        result[name] = variable
    return result


def _number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    return float(value)


def _percent(value: Any, *, context: str) -> float:
    if not isinstance(value, str) or not value.endswith("%"):
        raise ValueError(f"{context} must be a percentage")
    return float(value.removesuffix("%")) / 100.0


def _qualifiers(variable: dict[str, Any], spec: _TableSpec) -> tuple[Qualifier, ...]:
    raw_qualifiers = variable.get("qualifiers")
    if not isinstance(raw_qualifiers, list):
        raise ValueError("Dependent-variable qualifiers must be a list")
    by_name = {raw.get("name"): raw for raw in raw_qualifiers if isinstance(raw, dict)}
    reaction = by_name.get("RE")
    sqrt_s = by_name.get("SQRT(S)")
    if reaction is None or reaction.get("value") != spec.reaction:
        raise ValueError(f"Table {spec.number} has an unexpected reaction qualifier")
    if sqrt_s is None or str(sqrt_s.get("value")) != spec.sqrt_s_qualifier:
        raise ValueError(f"Table {spec.number} has an unexpected SQRT(S) qualifier")
    if sqrt_s.get("units") != "GeV":
        raise ValueError(f"Table {spec.number} has unexpected SQRT(S) units")

    output = [
        Qualifier("RE", spec.reaction),
        Qualifier("SQRT(S)", spec.sqrt_s_qualifier, "GeV"),
        Qualifier("OBSERVABLE", OBSERVABLE, OUTPUT_UNIT),
    ]
    if spec.fixed_photon_energy is not None:
        energy = by_name.get("E")
        if (
            energy is None
            or str(energy.get("value")) != str(spec.fixed_photon_energy)
            or energy.get("units") != "GeV"
        ):
            raise ValueError(f"Table {spec.number} has an unexpected E qualifier")
        output.insert(0, Qualifier("E", spec.fixed_photon_energy, "GeV"))
    return tuple(output)


def _array(variable: dict[str, Any], *, context: str) -> np.ndarray:
    return np.array(
        [_number(item.get("value"), context=context) for item in variable["values"]]
    )


def _parse_table(table: dict[str, Any], spec: _TableSpec) -> Measurement:
    independent = _variables_by_name(
        table, "independent_variables", expected_points=spec.points
    )
    dependent = _variables_by_name(
        table, "dependent_variables", expected_points=spec.points
    )
    if set(dependent) != {OBSERVABLE}:
        raise ValueError(f"Table {spec.number} has unexpected dependent variables")
    cross_section_variable = dependent[OBSERVABLE]
    if cross_section_variable["header"].get("units") != SOURCE_UNIT:
        raise ValueError(f"Table {spec.number} has unexpected cross-section units")
    if "THETA" not in independent:
        raise ValueError(f"Table {spec.number} is missing THETA")
    if independent["THETA"]["header"].get("units") != "DEG":
        raise ValueError(f"Table {spec.number} has unexpected THETA units")

    if spec.fixed_photon_energy is None:
        if set(independent) != {"E", "THETA"}:
            raise ValueError(f"Table {spec.number} has unexpected kinematic variables")
        if independent["E"]["header"].get("units") != "GEV":
            raise ValueError(f"Table {spec.number} has unexpected E units")
        photon_energy = _array(independent["E"], context=f"Table {spec.number} E")
    else:
        if set(independent) != {"THETA"}:
            raise ValueError(f"Table {spec.number} has unexpected kinematic variables")
        photon_energy = np.full(spec.points, spec.fixed_photon_energy)

    theta = np.deg2rad(
        _array(independent["THETA"], context=f"Table {spec.number} THETA")
    )
    q2 = np.zeros(spec.points)
    w = np.sqrt(MASS_PROTON**2 + 2.0 * MASS_PROTON * photon_energy)

    values = []
    point_errors = []
    systematic_fractions = []
    for row, item in enumerate(cross_section_variable["values"], start=1):
        values.append(
            _number(item.get("value"), context=f"Table {spec.number} row {row}")
        )
        errors = item.get("errors")
        if not isinstance(errors, list) or len(errors) != 2:
            raise ValueError(f"Table {spec.number} row {row} must have two errors")
        point_error, systematic = errors
        if point_error.get("label") is not None or "symerror" not in point_error:
            raise ValueError(f"Table {spec.number} row {row} has invalid point error")
        if systematic.get("label") != "sys":
            raise ValueError(
                f"Table {spec.number} row {row} has unexpected systematic label"
            )
        point_errors.append(
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

    cross_section = np.asarray(values) * _MUB_TO_NB
    point_error = np.asarray(point_errors) * _MUB_TO_NB
    systematic = cross_section * np.asarray(systematic_fractions)
    return Measurement(
        xsect=Xsect(
            sigU=cross_section,
            sigU_err=point_error,
            sigU_sys_err_low=systematic,
            sigU_sys_err_high=systematic,
        ),
        kinematics=Kinematics(Q2=q2, W=w, thetacm=theta),
        metadata=Metadata(
            reaction=spec.reaction,
            unit=OUTPUT_UNIT,
            source_unit=SOURCE_UNIT,
            angle_unit="rad",
            dataset_key=DATASET_KEY,
            doi=DOI,
            record_id=RECORD_ID,
            table_id=f"Table {spec.number}",
            observable=OBSERVABLE,
            experiment="Bonn",
            year=1970,
            reference=(
                "Bleckmann et al., Z. Phys. 239, 1-15 (1970); " "DOI:10.1007/BF01408507"
            ),
            qualifiers=_qualifiers(cross_section_variable, spec),
        ),
    )


def parse(input_dir: Path, output_dir: Path) -> None:
    """Parse four Bonn photoproduction tables into common JSON measurements."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    submission = _load_submission(input_dir / "submission.yaml")
    if submission.get("hepdata_doi") != DOI:
        raise ValueError(
            f"Expected HEPData DOI {DOI}, found {submission.get('hepdata_doi')!r}"
        )

    for spec in _TABLES:
        measurement = _parse_table(
            _load_yaml(input_dir / f"Table{spec.number}.yaml"), spec
        )
        write_json_atomic(output_dir / spec.output_name, measurement)
