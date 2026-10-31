from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from ep_data.models import (
    Kinematics,
    Measurement,
    Metadata,
    Qualifier,
    Xsect,
)


def test_xsect_accepts_any_component_and_fills_errors():
    xsect = Xsect(sigU=[1, 2])

    np.testing.assert_array_equal(xsect.sigU, [1.0, 2.0])
    np.testing.assert_array_equal(xsect.sigU_err, [0.0, 0.0])
    np.testing.assert_array_equal(xsect.sigU_sys_err_low, [0.0, 0.0])
    np.testing.assert_array_equal(xsect.sigU_sys_err_high, [0.0, 0.0])
    np.testing.assert_array_equal(xsect.sigU_lower_limit, [np.nan, np.nan])
    np.testing.assert_array_equal(xsect.sigU_upper_limit, [np.nan, np.nan])
    assert xsect.sigL is None
    assert xsect.sigL_err is None
    assert xsect.sigL_sys_err_low is None
    assert xsect.sigL_sys_err_high is None


def test_xsect_preserves_supplied_errors():
    xsect = Xsect(
        sigL=[1, 2],
        sigL_err=[0.1, 0.2],
        sigL_sys_err_low=[0.3, 0.4],
        sigL_sys_err_high=[0.5, 0.6],
        sigL_upper_limit=[np.nan, 3.0],
    )

    np.testing.assert_array_equal(xsect.sigL_err, [0.1, 0.2])
    np.testing.assert_array_equal(xsect.sigL_sys_err_low, [0.3, 0.4])
    np.testing.assert_array_equal(xsect.sigL_sys_err_high, [0.5, 0.6])
    np.testing.assert_array_equal(xsect.sigL_upper_limit, [np.nan, 3.0])


def test_xsect_rejects_empty_instance():
    with pytest.raises(ValueError, match="at least one"):
        Xsect()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigU": [[1, 2]]},
        {"sigU": [1, 2], "sigL": [1]},
        {"sigU": [1, 2], "sigU_err": [0.1]},
        {"sigU_err": [0.1]},
        {"sigU": [1, 2], "sigU_upper_limit": [3.0]},
        {"sigU_upper_limit": [3.0]},
    ],
)
def test_xsect_rejects_invalid_shapes_and_orphan_errors(kwargs):
    with pytest.raises(ValueError):
        Xsect(**kwargs)


def test_kinematics_t_is_optional_and_errors_default_to_zero():
    kinematics = Kinematics(
        Q2=[1, 2],
        W=[2, 3],
        thetacm=[10, np.nan],
        thetacm_center=[7.5, 15],
        epsilon=[0.8, 0.9],
    )

    assert kinematics.t is None
    assert kinematics.t_err is None
    np.testing.assert_array_equal(kinematics.Q2_err, [0.0, 0.0])
    np.testing.assert_array_equal(kinematics.W_err, [0.0, 0.0])
    np.testing.assert_array_equal(kinematics.epsilon_err, [0.0, 0.0])
    np.testing.assert_array_equal(kinematics.thetacm_center, [7.5, 15.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"Q2": [1, 2], "W": [2]},
        {"Q2": [[1, 2]], "W": [2, 3]},
        {"Q2": [1, 2], "W": [2, 3], "t_err": [0.1, 0.2]},
        {"Q2": [1, 2], "W": [2, 3], "Q2_err": [0.1]},
    ],
)
def test_kinematics_rejects_invalid_shapes_and_orphan_errors(kwargs):
    with pytest.raises(ValueError):
        Kinematics(**kwargs)


def test_metadata_is_frozen_and_normalizes_qualifiers():
    qualifier = Qualifier(name="SQRT(S)", value="1.9-2.8", unit="GeV")
    metadata = Metadata(
        reaction="GAMMA* P --> K+ LAMBDA",
        unit="nb/sr",
        source_unit="NB/SR",
        angle_unit="rad",
        qualifiers=[qualifier],
    )

    assert metadata.qualifiers == (qualifier,)
    with pytest.raises(FrozenInstanceError):
        metadata.unit = "ub/sr"


def test_common_objects_have_concise_string_representations():
    qualifier = Qualifier("W", 1.9, "GeV")
    metadata = Metadata(
        reaction="reaction",
        unit="nb/sr",
        angle_unit="rad",
        experiment="experiment",
        year=2020,
        qualifiers=(qualifier,),
    )
    xsect = Xsect(sigL=[1, 2], sigT=[3, 4])
    kinematics = Kinematics(Q2=[1, 2], W=[2, 3], thetacm=[0.1, 0.2])
    measurement = Measurement(xsect=xsect, kinematics=kinematics, metadata=metadata)

    assert str(qualifier) == "W=1.9 [GeV]"
    assert str(metadata) == (
        "Metadata(reaction='reaction', experiment='experiment', year=2020, "
        "unit='nb/sr', angle_unit='rad')"
    )
    assert str(xsect) == "Xsect(points=2, components=sigL,sigT)"
    assert str(kinematics) == "Kinematics(points=2, variables=Q2,W,thetacm)"
    assert str(measurement) == (
        "Measurement(reaction='reaction', points=2, components=sigL,sigT, "
        "experiment='experiment', year=2020)"
    )


def test_metadata_rejects_untyped_qualifiers():
    with pytest.raises(TypeError, match="Qualifier"):
        Metadata(reaction="reaction", qualifiers=[{"name": "W", "value": 2.0}])


def test_measurement_validates_types_and_length():
    measurement = Measurement(
        xsect=Xsect(sigT=[1, 2]),
        kinematics=Kinematics(Q2=[1, 2], W=[2, 3]),
        metadata=Metadata(reaction="reaction"),
    )
    assert measurement.metadata.reaction == "reaction"

    with pytest.raises(ValueError, match="match Xsect"):
        Measurement(
            xsect=Xsect(sigT=[1]),
            kinematics=Kinematics(Q2=[1, 2], W=[2, 3]),
            metadata=Metadata(reaction="reaction"),
        )

    with pytest.raises(TypeError, match="metadata"):
        Measurement(
            xsect=Xsect(sigT=[1]),
            kinematics=Kinematics(Q2=[1], W=[2]),
            metadata={},
        )
