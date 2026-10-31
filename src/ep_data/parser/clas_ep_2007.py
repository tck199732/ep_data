"""Parser for the approved response tables in HEPData record ins732363."""

from __future__ import annotations

import math
import re
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

DATASET_KEY = "clas_ep_2007"
DOI = "10.17182/hepdata.4994.v1"
RECORD_ID = "732363"
SOURCE_UNIT = "NB/STR"
OUTPUT_UNIT = "nb/sr"
REFERENCE = (
    "P. Ambrozewicz et al. (CLAS Collaboration), "
    "Phys. Rev. C 75, 045203 (2007); DOI:10.1103/PhysRevC.75.045203"
)

_UNSEPARATED = (
    ("SIG", "sigU"),
    ("SIG(C=TT)", "sigTT"),
    ("SIG(C=TL)", "sigLT"),
)
_SIMULTANEOUS = (
    ("SIG(C=L)", "sigL"),
    ("SIG(C=T)", "sigT"),
    ("SIG(C=TT)", "sigTT"),
    ("SIG(C=TL)", "sigLT"),
)
_ROSENBLUTH = (("SIG(C=L)", "sigL"), ("SIG(C=T)", "sigT"))
_RATIO = "SIG(C=L)/SIG(C=T)"
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_BIN_RE = re.compile(
    rf"^\s*(?P<center>{_NUMBER})\s*\(BIN\s*=\s*(?P<low>{_NUMBER})"
    rf"\s+TO\s+(?P<high>{_NUMBER})\s*\)\s*$",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(rf"^\s*(?P<low>{_NUMBER})\s*-\s*(?P<high>{_NUMBER})\s*$")


@dataclass(frozen=True, slots=True)
class _MeasurementSpec:
    output_name: str
    tables: tuple[int, ...]
    reaction: str
    beam_energy: float
    variables: tuple[tuple[str, str], ...]
    method: str


_MEASUREMENTS = (
    _MeasurementSpec(
        "lambda_e2p567_fine_w.json",
        tuple(range(1, 16)),
        "E- P --> E- K+ LAMBDA X",
        2.567,
        _UNSEPARATED,
        "phi fit; finer W binning",
    ),
    _MeasurementSpec(
        "lambda_e4.json",
        tuple(range(16, 46)),
        "E- P --> E- K+ LAMBDA X",
        4.0,
        _UNSEPARATED,
        "phi fit",
    ),
    _MeasurementSpec(
        "lambda_e2p567_coarse_w.json",
        tuple(range(46, 56)),
        "E- P --> E- K+ LAMBDA X",
        2.567,
        _UNSEPARATED,
        "phi fit; coarser W binning",
    ),
    _MeasurementSpec(
        "sigma0_e2p567_fine_w.json",
        tuple(range(56, 70)),
        "E- P --> E- K+ SIGMA0 X",
        2.567,
        _UNSEPARATED,
        "phi fit; finer W binning",
    ),
    _MeasurementSpec(
        "sigma0_e4.json",
        tuple(range(70, 77)) + tuple(range(84, 96)),
        "E- P --> E- K+ SIGMA0 X",
        4.0,
        _UNSEPARATED,
        "phi fit; duplicate Tables 77-83 omitted",
    ),
    _MeasurementSpec(
        "sigma0_e2p567_coarse_w.json",
        tuple(range(96, 104)),
        "E- P --> E- K+ SIGMA0 X",
        2.567,
        _UNSEPARATED,
        "phi fit; coarser W binning",
    ),
    _MeasurementSpec(
        "lambda_simultaneous_epsilon_phi.json",
        tuple(range(104, 108)),
        "E- P --> E- K+ LAMBDA X",
        2.567,
        _SIMULTANEOUS,
        "simultaneous epsilon-phi fit",
    ),
    _MeasurementSpec(
        "lambda_rosenbluth.json",
        tuple(range(108, 112)),
        "E- P --> E- K+ LAMBDA X",
        2.567,
        _ROSENBLUTH,
        "Rosenbluth separation; L/T ratio omitted",
    ),
    _MeasurementSpec(
        "sigma0_simultaneous_epsilon_phi.json",
        tuple(range(112, 115)),
        "E- P --> E- K+ SIGMA0 X",
        2.567,
        _SIMULTANEOUS,
        "simultaneous epsilon-phi fit",
    ),
    _MeasurementSpec(
        "sigma0_rosenbluth.json",
        tuple(range(115, 118)),
        "E- P --> E- K+ SIGMA0 X",
        2.567,
        _ROSENBLUTH,
        "Rosenbluth separation; L/T ratio omitted",
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
            first = next(yaml.safe_load_all(stream), None)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read {path.name}: {exc}") from exc
    if not isinstance(first, dict):
        raise ValueError(f"{path.name} must begin with a YAML mapping")
    return first


def _number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    return float(value)


def _qualifier_map(variable: dict[str, Any], table_number: int) -> dict[str, Any]:
    raw = variable.get("qualifiers")
    if not isinstance(raw, list):
        raise ValueError(f"Table {table_number} qualifiers must be a list")
    qualifiers: dict[str, Any] = {}
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError(f"Table {table_number} has an invalid qualifier")
        name = item["name"]
        if name in qualifiers:
            raise ValueError(f"Table {table_number} has duplicate qualifier {name}")
        qualifiers[name] = item
    return qualifiers


def _parse_bin(value: Any, context: str) -> tuple[float, float, float]:
    if not isinstance(value, str) or (match := _BIN_RE.fullmatch(value)) is None:
        raise ValueError(f"{context} has an invalid bin qualifier: {value!r}")
    return tuple(float(match.group(name)) for name in ("center", "low", "high"))


def _parse_range(value: Any, context: str) -> tuple[float, float]:
    if not isinstance(value, str) or (match := _RANGE_RE.fullmatch(value)) is None:
        raise ValueError(f"{context} has an invalid range qualifier: {value!r}")
    return float(match.group("low")), float(match.group("high"))


def _table_variables(
    table: dict[str, Any], spec: _MeasurementSpec, table_number: int
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    independent = table.get("independent_variables")
    dependent = table.get("dependent_variables")
    if not isinstance(independent, list) or len(independent) != 1:
        raise ValueError(f"Table {table_number} must have one independent variable")
    angle = independent[0]
    if not isinstance(angle, dict) or angle.get("header") != {
        "name": "COS(THETA(P=4))"
    }:
        raise ValueError(f"Table {table_number} has an unexpected angular variable")
    angle_values = angle.get("values")
    if not isinstance(angle_values, list):
        raise ValueError(f"Table {table_number} angular values must be a list")

    if not isinstance(dependent, list):
        raise ValueError(f"Table {table_number} dependent variables must be a list")
    by_name: dict[str, dict[str, Any]] = {}
    for variable in dependent:
        if not isinstance(variable, dict) or not isinstance(
            variable.get("header"), dict
        ):
            raise ValueError(f"Table {table_number} has an invalid dependent variable")
        name = variable["header"].get("name")
        if not isinstance(name, str) or name in by_name:
            raise ValueError(f"Table {table_number} has invalid dependent headers")
        by_name[name] = variable

    expected = {source for source, _ in spec.variables}
    if spec.variables == _ROSENBLUTH:
        expected.add(_RATIO)
    if set(by_name) != expected:
        raise ValueError(
            f"Table {table_number} has unexpected dependent variables: "
            f"{', '.join(sorted(by_name))}"
        )
    for name in expected.difference({_RATIO}):
        variable = by_name[name]
        if variable["header"].get("units") != SOURCE_UNIT:
            raise ValueError(f"Table {table_number} has unexpected units for {name}")
        values = variable.get("values")
        if not isinstance(values, list) or len(values) != len(angle_values):
            raise ValueError(f"Table {table_number} has mismatched values for {name}")
    return angle, by_name


def _table_kinematics(
    angle: dict[str, Any],
    variables: dict[str, dict[str, Any]],
    spec: _MeasurementSpec,
    table_number: int,
) -> tuple[list[float], list[float], list[float], str, str, list[str]]:
    qualifiers = _qualifier_map(variables[spec.variables[0][0]], table_number)
    if set(qualifiers) != {"E(P=1)", "Q**2", "RE", "SQRT(S)", "W"}:
        raise ValueError(f"Table {table_number} has unexpected qualifiers")
    for source_name, _ in spec.variables[1:]:
        if _qualifier_map(variables[source_name], table_number) != qualifiers:
            raise ValueError(f"Table {table_number} has inconsistent qualifiers")

    beam = qualifiers["E(P=1)"]
    reaction = qualifiers["RE"]
    q2_raw = qualifiers["Q**2"]
    w_raw = qualifiers["W"]
    sqrt_s = qualifiers["SQRT(S)"]
    try:
        beam_value = float(beam.get("value"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Table {table_number} has an invalid beam energy") from exc
    if beam.get("units") != "GeV" or beam_value != spec.beam_energy:
        raise ValueError(f"Table {table_number} has an unexpected beam energy")
    if reaction.get("value") != spec.reaction:
        raise ValueError(f"Table {table_number} has an unexpected reaction")
    if q2_raw.get("units") != "GEV**2" or w_raw.get("units") != "GEV":
        raise ValueError(f"Table {table_number} has unexpected kinematic units")
    if sqrt_s.get("units") != "GeV":
        raise ValueError(f"Table {table_number} has unexpected SQRT(S) units")

    q2_center, q2_low, q2_high = _parse_bin(
        q2_raw.get("value"), f"Table {table_number} Q**2"
    )
    w_center, w_low, w_high = _parse_bin(w_raw.get("value"), f"Table {table_number} W")
    sqrt_low, sqrt_high = _parse_range(
        sqrt_s.get("value"), f"Table {table_number} SQRT(S)"
    )
    if not (math.isclose(w_low, sqrt_low) and math.isclose(w_high, sqrt_high)):
        raise ValueError(f"Table {table_number} has inconsistent W qualifiers")

    centers = []
    cos_bins = []
    for row, item in enumerate(angle["values"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Table {table_number} angle row {row} is invalid")
        low = _number(item.get("low"), context=f"Table {table_number} cos low")
        high = _number(item.get("high"), context=f"Table {table_number} cos high")
        if not (-1.0 <= low < high <= 1.0):
            raise ValueError(f"Table {table_number} angle row {row} has invalid bounds")
        centers.append(math.acos((low + high) / 2.0))
        cos_bins.append(f"{low:g}:{high:g}")

    points = len(centers)
    return (
        [q2_center] * points,
        [w_center] * points,
        centers,
        f"{q2_low:g}:{q2_high:g}",
        f"{w_low:g}:{w_high:g}",
        cos_bins,
    )


def _is_sentinel(source_name: str, value: float, error: float) -> bool:
    sentinels = {
        "SIG": {0.0},
        "SIG(C=TT)": {0.0, 1.0},
        "SIG(C=TL)": {0.0, 1.41},
    }
    return error == 0.0 and value in sentinels.get(source_name, set())


def _response_values(
    variable: dict[str, Any], source_name: str, table_number: int
) -> tuple[list[float], list[float], list[float]]:
    central = []
    point_errors = []
    systematic_errors = []
    for row, item in enumerate(variable["values"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Table {table_number} {source_name} row {row} is invalid")
        value = _number(
            item.get("value"), context=f"Table {table_number} {source_name} value"
        )
        errors = item.get("errors")
        if not isinstance(errors, list):
            raise ValueError(f"Table {table_number} {source_name} errors are invalid")

        if source_name in {"SIG(C=L)", "SIG(C=T)"}:
            if len(errors) != 2:
                raise ValueError(
                    f"Table {table_number} {source_name} must have two errors"
                )
            by_label = {
                error.get("label"): error for error in errors if isinstance(error, dict)
            }
            if set(by_label) != {"stat", "sys"}:
                raise ValueError(
                    f"Table {table_number} {source_name} has unexpected errors"
                )
            point_error = _number(
                by_label["stat"].get("symerror"),
                context=f"Table {table_number} {source_name} stat error",
            )
            systematic = _number(
                by_label["sys"].get("symerror"),
                context=f"Table {table_number} {source_name} sys error",
            )
        else:
            if (
                len(errors) != 1
                or not isinstance(errors[0], dict)
                or errors[0].get("label") is not None
            ):
                raise ValueError(
                    f"Table {table_number} {source_name} has unexpected errors"
                )
            point_error = _number(
                errors[0].get("symerror"),
                context=f"Table {table_number} {source_name} error",
            )
            systematic = 0.0

        if _is_sentinel(source_name, value, point_error):
            value = point_error = systematic = math.nan
        central.append(value)
        point_errors.append(point_error)
        systematic_errors.append(systematic)
    return central, point_errors, systematic_errors


def _format_tables(tables: tuple[int, ...]) -> str:
    ranges = []
    start = previous = tables[0]
    for number in tables[1:]:
        if number != previous + 1:
            ranges.append((start, previous))
            start = number
        previous = number
    ranges.append((start, previous))
    return ", ".join(
        f"Table {low}" if low == high else f"Tables {low}-{high}"
        for low, high in ranges
    )


def _parse_measurement(input_dir: Path, spec: _MeasurementSpec) -> Measurement:
    values = {target: [] for _, target in spec.variables}
    errors = {target: [] for _, target in spec.variables}
    systematics = {target: [] for _, target in spec.variables}
    q2: list[float] = []
    w: list[float] = []
    theta_center: list[float] = []
    q2_bins: list[str] = []
    w_bins: list[str] = []
    cos_bins: list[str] = []

    for table_number in spec.tables:
        table = _load_yaml(input_dir / f"Table{table_number}.yaml")
        angle, variables = _table_variables(table, spec, table_number)
        table_q2, table_w, table_theta, q2_bin, w_bin, table_cos_bins = (
            _table_kinematics(angle, variables, spec, table_number)
        )
        q2.extend(table_q2)
        w.extend(table_w)
        theta_center.extend(table_theta)
        q2_bins.append(q2_bin)
        w_bins.append(w_bin)
        cos_bins.extend(table_cos_bins)

        for source_name, target_name in spec.variables:
            central, point_errors, systematic_errors = _response_values(
                variables[source_name], source_name, table_number
            )
            values[target_name].extend(central)
            errors[target_name].extend(point_errors)
            systematics[target_name].extend(systematic_errors)

    xsect_arguments: dict[str, np.ndarray] = {}
    for _, target_name in spec.variables:
        xsect_arguments[target_name] = np.asarray(values[target_name])
        xsect_arguments[f"{target_name}_err"] = np.asarray(errors[target_name])
        systematic = np.asarray(systematics[target_name])
        xsect_arguments[f"{target_name}_sys_err_low"] = systematic
        xsect_arguments[f"{target_name}_sys_err_high"] = systematic

    observables = ", ".join(source for source, _ in spec.variables)
    qualifiers = (
        Qualifier("E(P=1)", spec.beam_energy, "GeV"),
        Qualifier("RE", spec.reaction),
        Qualifier("OBSERVABLES", observables, OUTPUT_UNIT),
        Qualifier("EXTRACTION_METHOD", spec.method),
        Qualifier("Q**2 BINS", ",".join(dict.fromkeys(q2_bins)), "GeV**2"),
        Qualifier("W BINS", ",".join(dict.fromkeys(w_bins)), "GeV"),
        Qualifier("COS(THETA) BINS", ",".join(dict.fromkeys(cos_bins))),
    )
    return Measurement(
        xsect=Xsect(**xsect_arguments),
        kinematics=Kinematics(
            Q2=np.asarray(q2),
            W=np.asarray(w),
            thetacm=np.full(len(q2), np.nan),
            thetacm_center=np.asarray(theta_center),
        ),
        metadata=Metadata(
            reaction=spec.reaction,
            unit=OUTPUT_UNIT,
            source_unit=SOURCE_UNIT,
            angle_unit="rad",
            dataset_key=DATASET_KEY,
            doi=DOI,
            record_id=RECORD_ID,
            table_id=_format_tables(spec.tables),
            observable=observables,
            experiment="CLAS",
            year=2007,
            reference=REFERENCE,
            qualifiers=qualifiers,
        ),
    )


def parse(input_dir: Path, output_dir: Path) -> None:
    """Parse the approved CLAS response tables into ten JSON measurements."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    submission = _load_submission(input_dir / "submission.yaml")
    if submission.get("hepdata_doi") != DOI:
        raise ValueError(
            f"Expected HEPData DOI {DOI}, found {submission.get('hepdata_doi')!r}"
        )

    for spec in _MEASUREMENTS:
        write_json_atomic(
            output_dir / spec.output_name, _parse_measurement(input_dir, spec)
        )
