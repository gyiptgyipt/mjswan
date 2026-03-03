---
icon: octicons/arrow-switch-16
---

# Using mjlab

[mjlab](https://github.com/mujocolab/mjlab) is a reinforcement learning framework built on top of MuJoCo. mjswan can visualize mjlab environments directly — there is no need to export or convert anything. Pass the `MjSpec` from an mjlab `Scene` straight to `add_scene()`.

## Basic integration

```python
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg
import mjswan

builder = mjswan.Builder()
project = builder.add_project(name="mjlab Examples")

env_cfg = load_env_cfg("Mjlab-Velocity-Flat-Anymal-C")
env_cfg.scene.num_envs = 1           # single environment for the viewer
scene_obj = Scene(env_cfg.scene, device="cpu")

scene = project.add_scene(
    spec=scene_obj.spec,             # MjSpec from mjlab
    name="ANYmal C",
)

app = builder.build()
app.launch()
```

## Iterating over all tasks

If you want to expose every registered task as a separate scene:

```python
from mjlab.scene import Scene
from mjlab.tasks.registry import list_tasks, load_env_cfg
import mjswan

builder = mjswan.Builder()
project = builder.add_project(name="mjlab Tasks")

for task_id in list_tasks():
    env_cfg = load_env_cfg(task_id)
    env_cfg.scene.num_envs = 1
    scene_obj = Scene(env_cfg.scene, device="cpu")
    project.add_scene(spec=scene_obj.spec, name=task_id)

builder.build().launch()
```

## Adding a trained policy

After training with mjlab, export your policy to ONNX and attach it alongside the scene:

```python
import onnx

scene = project.add_scene(spec=scene_obj.spec, name="ANYmal C")

policy = scene.add_policy(
    policy=onnx.load("checkpoints/anymal_c.onnx"),
    name="velocity 3000 iters",
    config_path="checkpoints/anymal_c.json",  # observation/action config
)
policy.add_velocity_command(
    lin_vel_x=(-1.0, 1.0),
    lin_vel_y=(-1.0, 1.0),
    ang_vel_z=(-0.5, 0.5),
    default_lin_vel_x=0.5,
)
```

## Using mjlab_myosuite

If your mjlab environment includes MyoSuite tasks (via `mjlab_myosuite`), import it before calling `list_tasks()` to register the extra task IDs:

```python
import os
os.environ["MUJOCO_GL"] = "disable"  # headless rendering

import mjlab_myosuite  # registers MyoSuite tasks into mjlab
from mjlab.scene import Scene
from mjlab.tasks.registry import list_tasks, load_env_cfg
import mjswan

builder = mjswan.Builder()
project = builder.add_project(name="mjlab + MyoSuite")

for task_id in list_tasks():
    env_cfg = load_env_cfg(task_id)
    env_cfg.scene.num_envs = 1
    scene_obj = Scene(env_cfg.scene, device="cpu")
    project.add_scene(spec=scene_obj.spec, name=task_id)

builder.build().launch()
```
