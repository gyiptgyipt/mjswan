"""Tests for mjswan.command — SliderConfig, ButtonConfig, CommandGroupConfig, velocity_command.

Layer: L1 (pure Python, no MuJoCo/ONNX required).
"""

import mjswan
from mjswan.command import (
    ButtonConfig,
    CommandGroupConfig,
    SliderConfig,
    velocity_command,
)


class TestSliderConfig:
    def test_min_max_derived_from_range(self):
        s = SliderConfig(name="x", label="X", range=(-2.0, 3.0))
        assert s.min == -2.0
        assert s.max == 3.0

    def test_to_dict_has_correct_type(self):
        s = SliderConfig(name="x", label="X")
        assert s.to_dict()["type"] == "slider"

    def test_to_dict_includes_all_fields(self):
        s = SliderConfig(
            name="lin_vel_x",
            label="Forward Velocity",
            range=(-1.0, 1.0),
            default=0.5,
            step=0.05,
        )
        d = s.to_dict()
        assert d["name"] == "lin_vel_x"
        assert d["label"] == "Forward Velocity"
        assert d["min"] == -1.0
        assert d["max"] == 1.0
        assert d["default"] == 0.5
        assert d["step"] == 0.05

    def test_default_values(self):
        s = SliderConfig(name="x", label="X")
        assert s.default == 0.0
        assert s.step == 0.01
        assert s.range == (-1.0, 1.0)

    def test_slider_is_alias_for_slider_config(self):
        assert mjswan.Slider is SliderConfig


class TestButtonConfig:
    def test_to_dict_has_correct_type(self):
        b = ButtonConfig(name="reset", label="Reset")
        assert b.to_dict()["type"] == "button"

    def test_to_dict_includes_name_and_label(self):
        b = ButtonConfig(name="reset", label="Reset Simulation")
        d = b.to_dict()
        assert d["name"] == "reset"
        assert d["label"] == "Reset Simulation"

    def test_button_is_alias_for_button_config(self):
        assert mjswan.Button is ButtonConfig


class TestCommandGroupConfig:
    def test_to_dict_nests_inputs(self):
        group = CommandGroupConfig(
            name="velocity",
            inputs=[
                SliderConfig(name="x", label="X", range=(-1.0, 1.0)),
                ButtonConfig(name="reset", label="Reset"),
            ],
        )
        d = group.to_dict()
        assert len(d["inputs"]) == 2
        assert d["inputs"][0]["type"] == "slider"
        assert d["inputs"][1]["type"] == "button"

    def test_to_dict_with_no_inputs(self):
        group = CommandGroupConfig(name="empty")
        assert group.to_dict()["inputs"] == []


class TestVelocityCommand:
    def test_returns_group_named_velocity(self):
        cmd = velocity_command()
        assert cmd.name == "velocity"

    def test_has_exactly_three_sliders(self):
        cmd = velocity_command()
        assert len(cmd.inputs) == 3
        assert all(isinstance(inp, SliderConfig) for inp in cmd.inputs)

    def test_slider_names_are_canonical(self):
        cmd = velocity_command()
        names = [inp.name for inp in cmd.inputs]
        assert names == ["lin_vel_x", "lin_vel_y", "ang_vel_z"]

    def test_custom_ranges_applied(self):
        cmd = velocity_command(
            lin_vel_x=(-2.0, 2.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-3.0, 3.0),
        )
        sliders = {inp.name: inp for inp in cmd.inputs}
        assert sliders["lin_vel_x"].range == (-2.0, 2.0)
        assert sliders["lin_vel_y"].range == (-1.0, 1.0)
        assert sliders["ang_vel_z"].range == (-3.0, 3.0)

    def test_custom_defaults_applied(self):
        cmd = velocity_command(
            default_lin_vel_x=0.5,
            default_lin_vel_y=0.1,
            default_ang_vel_z=0.2,
        )
        sliders = {inp.name: inp for inp in cmd.inputs}
        assert sliders["lin_vel_x"].default == 0.5
        assert sliders["lin_vel_y"].default == 0.1
        assert sliders["ang_vel_z"].default == 0.2

    def test_velocity_command_is_accessible_from_mjswan(self):
        assert mjswan.velocity_command is velocity_command
