"""Tests for the shared box-retargeting geometry profiles."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from lm.box_config import (
    DEFAULT_TARGET_BOX_ORIENTATION_OFFSET_RPY_DEG,
    REAL_TARGET_BOX_GEOMETRY,
    SIM_TARGET_BOX_GEOMETRY,
    SOURCE_BOX_GEOMETRY,
    format_box_size_xyz,
    format_orientation_offset_rpy_deg,
    parse_box_size_xyz,
    parse_orientation_offset_rpy_deg,
)


def test_profiles_are_immutable_and_round_trip_through_ros_strings() -> None:
    for profile in (
        SOURCE_BOX_GEOMETRY,
        REAL_TARGET_BOX_GEOMETRY,
        SIM_TARGET_BOX_GEOMETRY,
    ):
        encoded = format_box_size_xyz(profile.size_xyz)
        np.testing.assert_allclose(parse_box_size_xyz(encoded), profile.size_xyz)

    with pytest.raises(FrozenInstanceError):
        SOURCE_BOX_GEOMETRY.forward_axis = "x"

    precise_size = (0.30025, 0.24975, 0.285125)
    np.testing.assert_allclose(
        parse_box_size_xyz(format_box_size_xyz(precise_size)),
        precise_size,
        rtol=0.0,
        atol=1e-12,
    )


def test_real_and_sim_profiles_are_selected_explicitly() -> None:
    assert REAL_TARGET_BOX_GEOMETRY is not SIM_TARGET_BOX_GEOMETRY
    assert SOURCE_BOX_GEOMETRY.forward_axis in {"x", "-x", "y", "-y", "z", "-z"}
    assert REAL_TARGET_BOX_GEOMETRY.up_axis in {"x", "-x", "y", "-y", "z", "-z"}
    assert SIM_TARGET_BOX_GEOMETRY.up_axis in {"x", "-x", "y", "-y", "z", "-z"}


def test_orientation_offset_default_is_zero_and_ros_string_round_trips() -> None:
    np.testing.assert_array_equal(
        DEFAULT_TARGET_BOX_ORIENTATION_OFFSET_RPY_DEG,
        np.zeros(3),
    )
    value = (-6.25, 9.5, 26.0)
    encoded = format_orientation_offset_rpy_deg(value)
    np.testing.assert_allclose(parse_orientation_offset_rpy_deg(encoded), value)
    np.testing.assert_allclose(
        parse_orientation_offset_rpy_deg("[-6.25, 9.5, 26]"),
        value,
    )


@pytest.mark.parametrize("value", ("1 2", "1 2 inf", [1.0, np.nan, 3.0]))
def test_orientation_offset_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError):
        parse_orientation_offset_rpy_deg(value)
