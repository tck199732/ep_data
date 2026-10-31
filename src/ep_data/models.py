"""Common structures produced by dataset-specific scientific parsers."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

_XSECT_COMPONENTS = ("sigU", "sigL", "sigT", "sigLT", "sigTT")
_KINEMATIC_VALUES = (
    "Q2",
    "W",
    "t",
    "t_center",
    "thetacm",
    "thetacm_center",
    "epsilon",
)
_KINEMATIC_ERRORS = {
    "Q2": "Q2_err",
    "W": "W_err",
    "t": "t_err",
    "thetacm": "thetacm_err",
    "epsilon": "epsilon_err",
}


def _convert_arrays(instance: object) -> None:
    for item in fields(instance):
        value = getattr(instance, item.name)
        if value is None:
            continue
        array = np.asarray(value, dtype=float)
        if array.ndim != 1:
            raise ValueError(f"{item.name} must be a one-dimensional array")
        setattr(instance, item.name, array)


def _validate_same_shape(
    group: str,
    values: list[tuple[str, np.ndarray | None]],
) -> tuple[int, ...]:
    present = [(name, value) for name, value in values if value is not None]
    if not present:
        raise ValueError(f"{group} must contain at least one array")

    expected_name, expected = present[0]
    expected_shape = expected.shape
    for name, value in present[1:]:
        if value.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {value.shape}; expected {expected_shape} "
                f"to match {expected_name}"
            )
    return expected_shape


@dataclass(frozen=True, slots=True)
class Qualifier:
    """An uninterpreted HEPData qualifier attached to a measurement."""

    name: str
    value: str | int | float
    unit: str | None = None

    def __str__(self) -> str:
        value = f"{self.name}={self.value}"
        return f"{value} [{self.unit}]" if self.unit is not None else value


@dataclass(frozen=True, slots=True)
class Metadata:
    """Provenance and unit information for one homogeneous measurement block."""

    reaction: str
    unit: str | None = None
    source_unit: str | None = None
    angle_unit: str | None = None
    dataset_key: str | None = None
    doi: str | None = None
    record_id: str | None = None
    table_id: str | None = None
    observable: str | None = None
    experiment: str | None = None
    year: int | None = None
    reference: str | None = None
    qualifiers: tuple[Qualifier, ...] = ()

    def __post_init__(self) -> None:
        qualifiers = tuple(self.qualifiers)
        if not all(isinstance(item, Qualifier) for item in qualifiers):
            raise TypeError("qualifiers must contain only Qualifier instances")
        object.__setattr__(self, "qualifiers", qualifiers)

    def __str__(self) -> str:
        details = [f"reaction={self.reaction!r}"]
        for name in ("experiment", "year", "observable", "unit", "angle_unit"):
            value = getattr(self, name)
            if value is not None:
                details.append(f"{name}={value!r}")
        return f"Metadata({', '.join(details)})"


@dataclass
class Xsect:
    """Cross-section components and their statistical and systematic errors."""

    sigU: np.ndarray | None = None
    sigL: np.ndarray | None = None
    sigT: np.ndarray | None = None
    sigLT: np.ndarray | None = None
    sigTT: np.ndarray | None = None
    sigU_err: np.ndarray | None = None
    sigL_err: np.ndarray | None = None
    sigT_err: np.ndarray | None = None
    sigLT_err: np.ndarray | None = None
    sigTT_err: np.ndarray | None = None
    sigU_sys_err_low: np.ndarray | None = None
    sigL_sys_err_low: np.ndarray | None = None
    sigT_sys_err_low: np.ndarray | None = None
    sigLT_sys_err_low: np.ndarray | None = None
    sigTT_sys_err_low: np.ndarray | None = None
    sigU_sys_err_high: np.ndarray | None = None
    sigL_sys_err_high: np.ndarray | None = None
    sigT_sys_err_high: np.ndarray | None = None
    sigLT_sys_err_high: np.ndarray | None = None
    sigTT_sys_err_high: np.ndarray | None = None
    sigU_lower_limit: np.ndarray | None = None
    sigL_lower_limit: np.ndarray | None = None
    sigT_lower_limit: np.ndarray | None = None
    sigLT_lower_limit: np.ndarray | None = None
    sigTT_lower_limit: np.ndarray | None = None
    sigU_upper_limit: np.ndarray | None = None
    sigL_upper_limit: np.ndarray | None = None
    sigT_upper_limit: np.ndarray | None = None
    sigLT_upper_limit: np.ndarray | None = None
    sigTT_upper_limit: np.ndarray | None = None

    def __post_init__(self) -> None:
        _convert_arrays(self)
        components = [(name, getattr(self, name)) for name in _XSECT_COMPONENTS]
        _validate_same_shape("Xsect", components)

        for name, value in components:
            error_names = (
                f"{name}_err",
                f"{name}_sys_err_low",
                f"{name}_sys_err_high",
            )
            limit_names = (f"{name}_lower_limit", f"{name}_upper_limit")
            related_names = error_names + limit_names
            if value is None:
                supplied = [
                    related_name
                    for related_name in related_names
                    if getattr(self, related_name) is not None
                ]
                if supplied:
                    raise ValueError(
                        f"{', '.join(supplied)} supplied without corresponding {name}"
                    )
                continue

            for related_name in related_names:
                related = getattr(self, related_name)
                if related is None:
                    default = (
                        np.zeros_like(value)
                        if related_name in error_names
                        else np.full_like(value, np.nan)
                    )
                    setattr(self, related_name, default)
                elif related.shape != value.shape:
                    raise ValueError(
                        f"{related_name} has shape {related.shape}; "
                        f"expected {value.shape} to match {name}"
                    )

    def __str__(self) -> str:
        components = [
            name for name in _XSECT_COMPONENTS if getattr(self, name) is not None
        ]
        points = getattr(self, components[0]).size
        return f"Xsect(points={points}, components={','.join(components)})"


@dataclass
class Kinematics:
    """Pointwise kinematics and optional uncertainties."""

    Q2: np.ndarray
    W: np.ndarray
    t: np.ndarray | None = None
    t_center: np.ndarray | None = None
    thetacm: np.ndarray | None = None
    thetacm_center: np.ndarray | None = None
    epsilon: np.ndarray | None = None
    Q2_err: np.ndarray | None = None
    W_err: np.ndarray | None = None
    thetacm_err: np.ndarray | None = None
    epsilon_err: np.ndarray | None = None
    t_err: np.ndarray | None = None

    def __post_init__(self) -> None:
        _convert_arrays(self)
        values = [(name, getattr(self, name)) for name in _KINEMATIC_VALUES]
        _validate_same_shape("Kinematics", values)

        for name, error_name in _KINEMATIC_ERRORS.items():
            value = getattr(self, name)
            error = getattr(self, error_name)
            if value is None:
                if error is not None:
                    raise ValueError(
                        f"{error_name} supplied without corresponding {name}"
                    )
                continue
            if error is None:
                setattr(self, error_name, np.zeros_like(value))
            elif error.shape != value.shape:
                raise ValueError(
                    f"{error_name} has shape {error.shape}; "
                    f"expected {value.shape} to match {name}"
                )

    def __str__(self) -> str:
        variables = [
            name for name in _KINEMATIC_VALUES if getattr(self, name) is not None
        ]
        return f"Kinematics(points={self.Q2.size}, " f"variables={','.join(variables)})"


@dataclass
class Measurement:
    """One reaction and one homogeneous block of cross sections and kinematics."""

    xsect: Xsect
    kinematics: Kinematics
    metadata: Metadata

    def __post_init__(self) -> None:
        if not isinstance(self.xsect, Xsect):
            raise TypeError("xsect must be an Xsect instance")
        if not isinstance(self.kinematics, Kinematics):
            raise TypeError("kinematics must be a Kinematics instance")
        if not isinstance(self.metadata, Metadata):
            raise TypeError("metadata must be a Metadata instance")

        xsect_shape = next(
            getattr(self.xsect, name).shape
            for name in _XSECT_COMPONENTS
            if getattr(self.xsect, name) is not None
        )
        if self.kinematics.Q2.shape != xsect_shape:
            raise ValueError(
                f"Kinematics has shape {self.kinematics.Q2.shape}; "
                f"expected {xsect_shape} to match Xsect"
            )

    def __str__(self) -> str:
        components = [
            name for name in _XSECT_COMPONENTS if getattr(self.xsect, name) is not None
        ]
        details = [
            f"reaction={self.metadata.reaction!r}",
            f"points={self.kinematics.Q2.size}",
            f"components={','.join(components)}",
        ]
        if self.metadata.experiment is not None:
            details.append(f"experiment={self.metadata.experiment!r}")
        if self.metadata.year is not None:
            details.append(f"year={self.metadata.year}")
        return f"Measurement({', '.join(details)})"
