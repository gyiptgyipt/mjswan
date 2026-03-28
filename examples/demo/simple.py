"""Simple mjswan Demo

A basic example demonstrating how to use mjswan to create a viewer application
with multiple robot scenes (Go2, Go1, and G1).
"""

import os
from pathlib import Path

import mujoco
import onnx
from mjlab.envs.mdp import observations as obs_fns
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

import mjswan


# mjlab has no simple_velocity_command. The mjswan adapter maps obs functions by
# __name__, so a local callable with the correct __name__ resolves to the
# SimpleVelocityCommand TS class and works with add_velocity_command().
def simple_velocity_command(*args, **kwargs): ...


# G1 humanoid: per-joint action scale, stiffness, damping (keyed by joint name)
_G1_SCALE = {
    "left_hip_pitch_joint": 0.5475464629911068,
    "left_hip_roll_joint": 0.35066146637882434,
    "left_hip_yaw_joint": 0.5475464629911068,
    "left_knee_joint": 0.35066146637882434,
    "left_ankle_pitch_joint": 0.43857731392336724,
    "left_ankle_roll_joint": 0.43857731392336724,
    "right_hip_pitch_joint": 0.5475464629911068,
    "right_hip_roll_joint": 0.35066146637882434,
    "right_hip_yaw_joint": 0.5475464629911068,
    "right_knee_joint": 0.35066146637882434,
    "right_ankle_pitch_joint": 0.43857731392336724,
    "right_ankle_roll_joint": 0.43857731392336724,
    "waist_yaw_joint": 0.5475464629911068,
    "waist_roll_joint": 0.43857731392336724,
    "waist_pitch_joint": 0.43857731392336724,
    "left_shoulder_pitch_joint": 0.43857731392336724,
    "left_shoulder_roll_joint": 0.43857731392336724,
    "left_shoulder_yaw_joint": 0.43857731392336724,
    "left_elbow_joint": 0.43857731392336724,
    "left_wrist_roll_joint": 0.43857731392336724,
    "left_wrist_pitch_joint": 0.07450087032950714,
    "left_wrist_yaw_joint": 0.07450087032950714,
    "right_shoulder_pitch_joint": 0.43857731392336724,
    "right_shoulder_roll_joint": 0.43857731392336724,
    "right_shoulder_yaw_joint": 0.43857731392336724,
    "right_elbow_joint": 0.43857731392336724,
    "right_wrist_roll_joint": 0.43857731392336724,
    "right_wrist_pitch_joint": 0.07450087032950714,
    "right_wrist_yaw_joint": 0.07450087032950714,
}


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
            # mjlab's JointPositionActionCfg does not have stiffness/damping fields;
            # those are mjswan-specific and are omitted here.
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=_G1_SCALE,
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
                    # mjlab has no simple_velocity_command; using generated_commands
                    # as a stand-in for adapter debugging.
                    "velocity_cmd": ObservationTermCfg(func=simple_velocity_command),
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
        # model=mujoco.MjModel.from_xml_path("assets/unitree_go2/scene.xml"),
        spec=mujoco.MjSpec.from_file("assets/unitree_go2/scene.xml"),
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
