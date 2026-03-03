---
icon: octicons/code-16
---

# Examples

Copy-paste patterns for common use cases. For a step-by-step first run, see [Quickstart](quickstart.md).

## Scene from an XML string

```python
import mujoco
import mjswan

builder = mjswan.Builder()
project = builder.add_project(name="Demo")

spec = mujoco.MjSpec.from_string("""
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
""")
project.add_scene(spec=spec, name="Sphere")

builder.build().launch()
```

## Scene from a file

```python
import mujoco
import mjswan

builder = mjswan.Builder()
project = builder.add_project(name="Robot")
project.add_scene(
    spec=mujoco.MjSpec.from_file("robot/scene.xml"),
    name="My Robot",
)
builder.build().launch()
```

## Policy with velocity command sliders

```python
import mujoco
import onnx
import mjswan

builder = mjswan.Builder()
project = builder.add_project(name="Robot")

scene = project.add_scene(
    spec=mujoco.MjSpec.from_file("robot/scene.xml"),
    name="G1",
)
scene.add_policy(
    policy=onnx.load("robot/locomotion.onnx"),
    name="Locomotion",
    config_path="robot/locomotion.json",
).add_velocity_command(
    lin_vel_x=(-1.5, 1.5),
    lin_vel_y=(-0.5, 0.5),
    default_lin_vel_x=0.5,
)

builder.build().launch()
```

`config_path` points to a JSON file describing the observation and action convention. See [Policy Config Format](../notes/policy-config.md) for the schema.

## Multiple policies on one scene

```python
scene = project.add_scene(spec=spec, name="Go2")

scene.add_policy(
    policy=onnx.load("policy_a.onnx"),
    name="Policy A",
    config_path="policy_a.json",
).add_velocity_command()

scene.add_policy(
    policy=onnx.load("policy_b.onnx"),
    name="Policy B",
    config_path="policy_b.json",
).add_velocity_command()
```

The browser UI shows a selector for choosing between policies at runtime.

## Custom command inputs

```python
policy.add_command(
    name="velocity",
    inputs=[
        mjswan.Slider("lin_vel_x", "Forward",  range=(-2.0, 2.0), default=0.5, step=0.05),
        mjswan.Slider("lin_vel_y", "Lateral",  range=(-0.5, 0.5), default=0.0, step=0.05),
        mjswan.Slider("ang_vel_z", "Yaw Rate", range=(-1.0, 1.0), default=0.0, step=0.05),
    ],
)
```

## Multiple projects

Each `add_project()` call maps to its own URL. The first project is always the root (`/`); the rest live at `/<id>/`.

```python
builder = mjswan.Builder(base_path="/demo/")

quadrupeds = builder.add_project(name="Quadrupeds")
quadrupeds.add_scene(spec=mujoco.MjSpec.from_file("go2/scene.xml"), name="Go2")
quadrupeds.add_scene(spec=mujoco.MjSpec.from_file("go1/scene.xml"), name="Go1")

humanoids = builder.add_project(name="Humanoids", id="humanoid")
humanoids.add_scene(spec=mujoco.MjSpec.from_file("g1/scene.xml"), name="G1")

builder.build().launch()
```

## Headless build (no browser)

```python
import os
os.environ["MJSWAN_NO_LAUNCH"] = "1"

app = builder.build()
# dist/ is ready; app.launch() is not called
```

Or without modifying the script:

```bash
MJSWAN_NO_LAUNCH=1 python build.py
```

See [Deployment](../guides/deployment.md) for GitHub Pages and CI/CD setup.
