---
icon: octicons/light-bulb-16
---

# Core Concepts

mjswan uses a four-level hierarchy to describe a browser application: **Builder → Project → Scene → Policy**. Understanding this structure is the fastest way to get oriented before looking at the API.

```
Builder
  └── Project
        └── Scene
              ├── Policy  (optional)
              └── Splat   (optional)
```

## Builder

`Builder` is the entry point. It collects everything you define and, when you call `build()`, compiles it into a self-contained web application written to `dist/` (or a directory you choose).

```python
import mjswan

builder = mjswan.Builder()
# … add projects …
app = builder.build()
app.launch()
```

Two optional constructor arguments matter for deployment:

| Argument | Default | Purpose |
|---|---|---|
| `base_path` | `"/"` | URL prefix when hosting at a subdirectory (e.g. `"/mjswan/"` for a GitHub Pages project page) |
| `gtm_id` | `None` | Google Tag Manager container ID; injects the GTM snippet when set |

## Project

A project groups related scenes under a single URL. The first project added becomes the root (`/`); additional projects are reachable at `/<project-id>/`.

```python
project = builder.add_project(name="My Robots")

# Explicit URL slug — accessible at /demo/
demo = builder.add_project(name="Demo", id="demo")
```

If you omit `id`, mjswan derives it automatically from the project name (spaces and hyphens become underscores, lowercased). The first project is always the root regardless of `id`.

## Scene

A scene contains exactly one MuJoCo model. You supply either an `MjSpec` or an `MjModel`:

```python
import mujoco

# Compressed .mjz — smaller output, recommended for large deployments
scene = project.add_scene(
    spec=mujoco.MjSpec.from_file("robot/scene.xml"),
    name="My Robot",
)

# Binary .mjb — loads slightly faster in the browser, produces larger files
scene = project.add_scene(
    model=mujoco.MjModel.from_xml_path("robot/scene.xml"),
    name="My Robot",
)
```

!!! tip "Which format should I use?"
    Use `spec=` unless you have a specific reason to prefer `model=`. The `.mjz` format uses DEFLATE compression and is significantly smaller — important when approaching GitHub Pages' 1 GB deployment limit.

## Policy

A policy is an ONNX model that runs inference inside the browser. Attach one or more policies to a scene:

```python
import onnx

policy = scene.add_policy(
    policy=onnx.load("locomotion.onnx"),
    name="Locomotion",
    config_path="locomotion.json",  # optional: observation/action config
)
```

Policies are purely client-side: inference runs in the browser via onnxruntime-web, so no server is needed at runtime.

### Commands

Commands let users interact with a running policy — for example, steering a walking robot with velocity sliders. Add them to a `PolicyHandle`:

```python
policy.add_command(
    name="velocity",
    inputs=[
        mjswan.Slider("lin_vel_x", "Forward Velocity", range=(-1.0, 1.0), default=0.5),
        mjswan.Slider("lin_vel_y", "Lateral Velocity", range=(-0.5, 0.5), default=0.0),
        mjswan.Slider("ang_vel_z",  "Yaw Rate",         range=(-1.0, 1.0), default=0.0),
    ],
)
```

For locomotion policies the convenience helper `add_velocity_command()` does the same thing with sensible defaults:

```python
policy.add_velocity_command(
    lin_vel_x=(-2.0, 2.0),
    default_lin_vel_x=0.5,
)
```

Available command inputs:

| Class | Description |
|---|---|
| `mjswan.Slider` | Continuous range slider. Fields: `name`, `label`, `range`, `default`, `step` |
| `mjswan.Button` | Momentary button. Fields: `name`, `label` |

## Output structure

`builder.build()` writes the following layout to the output directory:

```
dist/
├── index.html
├── assets/
│   ├── config.json          ← project/scene/policy manifest
│   └── …                    ← compiled JS/CSS
└── <project-id>/            ← "main" for the first project
    ├── index.html
    └── assets/
        └── <scene-id>/
            ├── scene.mjz    ← or scene.mjb
            ├── <policy>.onnx
            └── <policy>.json
```

The result is a fully static site: copy `dist/` to any static host (GitHub Pages, Netlify, S3, …) and it works without a server.

<!-- MEDIA: suggest a screenshot of the browser UI showing scene and policy selector panels -->

## Environment variables

| Variable | Effect |
|---|---|
| `MJSWAN_BASE_PATH` | Overrides `base_path` at build time (e.g. in CI pipelines) |
| `MJSWAN_NO_LAUNCH` | Set to `"1"` to suppress `app.launch()` opening a browser — useful in headless or Colab environments |
