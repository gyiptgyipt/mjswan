"""mjlab Integration Example - Visualize MuJoCo scenes from all mjlab default tasks

Extracts the MuJoCo model from each mjlab default task and visualizes them
in the browser using mjswan.
"""

from mjlab.tasks.registry import (  # noqa: F401 - for task registrations
    _REGISTRY,
    load_env_cfg,
)

import mjswan
from mjswan import ViewerConfig

ENTITY = "ttktjmt-org"
PROJECT = "mjlab"
TASK_RUN_ID_MAP = {
    # "Mjlab-Cartpole-Balance": "cartpole-balance",
    # "Mjlab-Cartpole-Swingup": "cartpole-swingup",
    # "Mjlab-Lift-Cube-Yam": "lift-cube-yam",
    "Mjlab-Velocity-Flat-Unitree-G1": "vel-flat-g1",
    "Mjlab-Velocity-Flat-Unitree-Go1": "vel-flat-go1-v3",
    # "Mjlab-Velocity-Rough-Unitree-G1": "vel-rough-g1",
    # "Mjlab-Velocity-Rough-Unitree-Go1": "vel-rough-go1",
}


def main():
    builder = mjswan.Builder()
    project = builder.add_project(name="mjlab Examples")

    for task_id, run_id in TASK_RUN_ID_MAP.items():
        env_cfg = load_env_cfg(task_id)
        scene = project.add_mjlab_scene(task_id)
        scene = scene.set_viewer_config(
            ViewerConfig(
                lookat=(0.0, 0.0, 0.4),
                distance=3,
                elevation=-20,
                azimuth=30,
            )
        )
        policies = scene.add_policy_from_wandb(
            f"{ENTITY}/{PROJECT}/{run_id}",
            task_id=task_id,
            observations={"policy": env_cfg.observations["actor"]},
            actions=env_cfg.actions,
            terminations=env_cfg.terminations,
        )
        for p in policies:
            p.add_velocity_command(name="twist")

    app = builder.build()
    app.launch()


if __name__ == "__main__":
    main()
