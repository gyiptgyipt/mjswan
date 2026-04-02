"""mjlab Integration Example - Visualize MuJoCo scenes from all mjlab default tasks

Extracts the MuJoCo model from each mjlab default task and visualizes them
in the browser using mjswan.
"""

from mjlab.tasks.registry import (  # noqa: F401 - for task registrations
    _REGISTRY,
    load_env_cfg,
)

import mjswan

ENTITY = "ttktjmt-org"
PROJECT = "mjlab"
TASK_RUN_ID_MAP = {
    # "Mjlab-Lift-Cube-Yam": "",
    # "Mjlab-Lift-Cube-Yam-Depth": "",
    # "Mjlab-Lift-Cube-Yam-Rgb": "",
    "Mjlab-Velocity-Flat-Unitree-G1": "vel-flat-g1",
    "Mjlab-Velocity-Flat-Unitree-Go1": "vel-flat-go1-v3",
    # "Mjlab-Velocity-Rough-Unitree-G1": "",
    # "Mjlab-Velocity-Rough-Unitree-Go1": "",
}


def main():
    builder = mjswan.Builder()
    project = builder.add_project(name="mjlab Examples")

    for task_id, run_id in TASK_RUN_ID_MAP.items():
        env_cfg = load_env_cfg(task_id)
        scene = project.add_mjlab_scene(task_id)
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
