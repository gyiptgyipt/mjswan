"""Simple mjswan Demo

A basic example demonstrating how to use mjswan to create a viewer application
with multiple robot scenes (Go2, Go1, and G1).
"""

import os
from pathlib import Path

import mujoco
import onnx

import mjswan
from mjswan.envs.mdp import observations as obs_fns
from mjswan.envs.mdp.actions import JointPositionActionCfg
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

# G1 humanoid: per-joint action scale, stiffness, damping
_G1_ACTION_SCALE = [
    0.5475464629911068,
    0.35066146637882434,
    0.5475464629911068,
    0.35066146637882434,
    0.43857731392336724,
    0.43857731392336724,
    0.5475464629911068,
    0.35066146637882434,
    0.5475464629911068,
    0.35066146637882434,
    0.43857731392336724,
    0.43857731392336724,
    0.5475464629911068,
    0.43857731392336724,
    0.43857731392336724,
    0.43857731392336724,
    0.43857731392336724,
    0.43857731392336724,
    0.43857731392336724,
    0.43857731392336724,
    0.07450087032950714,
    0.07450087032950714,
    0.43857731392336724,
    0.43857731392336724,
    0.43857731392336724,
    0.43857731392336724,
    0.43857731392336724,
    0.07450087032950714,
    0.07450087032950714,
]
_G1_STIFFNESS = [
    40.17923863450712,
    99.09842777666111,
    40.17923863450712,
    99.09842777666111,
    28.50124619574858,
    28.50124619574858,
    40.17923863450712,
    99.09842777666111,
    40.17923863450712,
    99.09842777666111,
    28.50124619574858,
    28.50124619574858,
    40.17923863450712,
    28.50124619574858,
    28.50124619574858,
    14.25062309787429,
    14.25062309787429,
    14.25062309787429,
    14.25062309787429,
    14.25062309787429,
    16.77832748089279,
    16.77832748089279,
    14.25062309787429,
    14.25062309787429,
    14.25062309787429,
    14.25062309787429,
    14.25062309787429,
    16.77832748089279,
    16.77832748089279,
]
_G1_DAMPING = [
    2.557889775413375,
    6.308801853496639,
    2.557889775413375,
    6.308801853496639,
    1.814445686584846,
    1.814445686584846,
    2.557889775413375,
    6.308801853496639,
    2.557889775413375,
    6.308801853496639,
    1.814445686584846,
    1.814445686584846,
    2.557889775413375,
    1.814445686584846,
    1.814445686584846,
    0.907222843292423,
    0.907222843292423,
    0.907222843292423,
    0.907222843292423,
    0.907222843292423,
    1.06814150219,
    1.06814150219,
    0.907222843292423,
    0.907222843292423,
    0.907222843292423,
    0.907222843292423,
    0.907222843292423,
    1.06814150219,
    1.06814150219,
]


def setup_builder() -> mjswan.Builder:
    """Set up and return the builder with demo projects configured.

    Creates a builder and adds a project with three robot scenes.
    Does not build or launch the application.

    Returns:
        Configured Builder instance ready to be built.
    """
    # Ensure asset-relative paths resolve regardless of current working directory.
    os.chdir(Path(__file__).resolve().parent)
    base_path = os.getenv("MJSWAN_BASE_PATH", "/")
    builder = mjswan.Builder(base_path=base_path)

    demo_project = builder.add_project(
        name="mjswan Demo",
    )

    demo_project.add_scene(
        spec=mujoco.MjSpec.from_file("assets/unitree_g1/scene.xml"),
        name="G1",
    ).set_viewer_config(
        mjswan.ViewerConfig(
            lookat=(0.0, 0.0, 0.7),
            distance=3.7,
            elevation=-13.0,
            azimuth=-34.0,
            origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
            body_name="torso_link",
        )
    ).add_policy(
        policy=onnx.load("assets/unitree_g1/locomotion.onnx"),
        name="Locomotion",
        config_path="assets/unitree_g1/locomotion.json",
        actions={
            "joint_pos": JointPositionActionCfg(
                scale=_G1_ACTION_SCALE,
                stiffness=_G1_STIFFNESS,
                damping=_G1_DAMPING,
            ),
        },
        observations={
            "policy": ObservationGroupCfg(
                terms={
                    "base_lin_vel": ObservationTermCfg(func=obs_fns.base_lin_vel),
                    "base_ang_vel": ObservationTermCfg(func=obs_fns.base_ang_vel),
                    "projected_gravity": ObservationTermCfg(
                        func=obs_fns.projected_gravity
                    ),
                    "joint_pos": ObservationTermCfg(
                        func=obs_fns.joint_pos_rel, params={"pos_steps": [0]}
                    ),
                    "joint_vel": ObservationTermCfg(func=obs_fns.joint_vel_rel),
                    "last_action": ObservationTermCfg(func=obs_fns.last_action),
                    "velocity_cmd": ObservationTermCfg(
                        func=obs_fns.simple_velocity_command
                    ),
                }
            )
        },
    ).add_velocity_command(
        lin_vel_x=(-2.0, 2.0),
        lin_vel_y=(-0.5, 0.5),
        default_lin_vel_x=0.5,
        default_lin_vel_y=0.0,
    )
    demo_project.add_scene(
        model=mujoco.MjModel.from_xml_path("assets/unitree_go2/scene.xml"),
        name="Go2",
    )

    return builder


def main():
    """Main entry point for the simple demo.

    Sets up the builder, builds the application, and launches it in a browser.

    Environment variables:
        MJSWAN_BASE_PATH: Base path for deployment (default: '/')
        MJSWAN_NO_LAUNCH: Set to '1' to skip launching the browser
    """
    builder = setup_builder()
    # Build and launch the application
    app = builder.build()
    if os.getenv("MJSWAN_NO_LAUNCH") == "1":
        return
    app.launch()


if __name__ == "__main__":
    main()
