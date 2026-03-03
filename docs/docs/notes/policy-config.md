---
icon: octicons/file-code-16
---

# Policy Config Format

When you call `scene.add_policy(..., config_path="policy.json")`, mjswan reads that JSON file, merges any command definitions into it, and writes the result alongside the ONNX file in `dist/`. The browser loads this file at runtime to know how to construct observations and apply actions.

This page documents the JSON schema used by the bundled examples. All fields are optional from mjswan's perspective — mjswan copies the file verbatim (merging only the `commands` key). The browser-side runtime is what interprets each field.

<!-- TODO: verify — the exact set of supported obs_config component names is defined in the browser runtime, not in the Python package. The fields below are derived from the bundled example configs. -->

## Top-level structure

```json
{
  "policy_joint_names": ["joint_1", "joint_2", "..."],
  "default_joint_pos":  [0.0, 0.4, -0.8, "..."],
  "control_type":       "joint_position",
  "action_scale":       0.5,
  "stiffness":          40.0,
  "damping":            1.0,
  "obs_config":         { "...": "..." },
  "onnx":               { "...": "..." },
  "commands":           { "...": "..." }
}
```

| Key | Type | Description |
|---|---|---|
| `policy_joint_names` | `string[]` | Ordered list of joint names the policy controls. Must match the order the ONNX model expects. |
| `default_joint_pos` | `number[]` | Default (resting) joint positions in radians, one per entry in `policy_joint_names`. Used when `subtract_default: true` in an observation component. |
| `control_type` | `string` | Action interpretation. `"joint_position"` is the only documented value; the output action is treated as a target joint position. |
| `action_scale` | `number \| number[]` | Scalar or per-joint scale applied to the raw ONNX output before sending to the PD controller. |
| `stiffness` | `number \| number[]` | PD controller position gain (Kp). Scalar applies to all joints; array sets per-joint values. |
| `damping` | `number \| number[]` | PD controller velocity gain (Kd). Same shape rules as `stiffness`. |
| `obs_config` | `object` | Observation groups keyed by their ONNX input tensor name. See [Observation config](#observation-config). |
| `onnx` | `object` | ONNX runtime metadata. See [ONNX metadata](#onnx-metadata). |
| `commands` | `object` | Injected automatically by mjswan when you call `add_command()` or `add_velocity_command()`. Do not write this key manually — it will be overwritten at build time. |

## Observation config

`obs_config` maps ONNX input tensor names to arrays of observation components. Each component is an object with a `name` field and optional parameters.

```json
"obs_config": {
  "policy": [
    { "name": "BaseAngularVelocity" },
    { "name": "ProjectedGravity", "joint_name": "floating_base_joint" },
    { "name": "JointPositions", "joint_names": "isaac", "subtract_default": true, "scale": 1.0 },
    { "name": "JointVelocities", "joint_names": "isaac", "scale": 0.05 },
    { "name": "PreviousActions" },
    { "name": "SimpleVelocityCommand" }
  ]
}
```

The key (`"policy"`, `"observation"`, `"obs"`, etc.) must match the corresponding `in_keys` entry in the `onnx.meta` block.

### Observation component fields

| Field | Default | Description |
|---|---|---|
| `name` | — | Component type. See [Component types](#component-types). |
| `history_steps` | `1` | Number of consecutive timesteps to stack. Set to `>1` for recurrent-style inputs. |
| `interleaved` | `false` | When `history_steps > 1`: if `true`, history is interleaved across components rather than concatenated per-component. |
| `scale` | `1.0` | Scalar multiplied into the component output before concatenation. |
| `subtract_default` | `false` | Subtract `default_joint_pos` from joint positions before outputting. |
| `joint_names` | — | `"isaac"` uses the order defined in `policy_joint_names`. Other values may be robot-specific. |
| `joint_name` | — | Single joint name, used by components that reference a specific joint (e.g. `ProjectedGravity`). |
| `world_frame` | `false` | When `true`, express velocities in the world frame instead of the base frame. |

### Component types

The following component names appear in the bundled examples:

| Name | Output | Notes |
|---|---|---|
| `BaseLinearVelocity` | 3D linear velocity of the base link | |
| `BaseAngularVelocity` | 3D angular velocity of the base link | |
| `ProjectedGravity` / `ProjectedGravityB` | Gravity vector projected into the base frame (3D) | Requires `joint_name` pointing to the floating base joint |
| `JointPositions` / `JointPos` | Joint positions for `policy_joint_names` joints | |
| `JointVelocities` | Joint velocities for `policy_joint_names` joints | |
| `PreviousActions` / `PrevActions` | Action output from the previous timestep | |
| `SimpleVelocityCommand` | lin_vel_x, lin_vel_y, ang_vel_z from the `velocity` command group | Requires a velocity command to be defined |
| `ImpedanceCommand` | Full impedance command vector | Used by impedance-control policies |

## ONNX metadata

The `onnx` block tells the browser runtime how to call the model:

```json
"onnx": {
  "path": "locomotion.onnx",
  "meta": {
    "in_keys":  ["policy"],
    "out_keys": ["action"],
    "in_shapes": [[[1, 45]]]
  }
}
```

| Key | Description |
|---|---|
| `path` | Filename of the ONNX file relative to the config file. Written automatically by mjswan at build time; you can omit this from your source JSON. |
| `meta.in_keys` | List of ONNX input tensor names, matching the keys in `obs_config`. |
| `meta.out_keys` | List of ONNX output tensor names. The tensor named `"action"` is used as the PD controller target. |
| `meta.in_shapes` | Optional explicit input shapes `[[[batch, dim], ...]]`. Required when the model has multiple inputs with ambiguous shapes (e.g. recurrent policies with a hidden state). |

## History and recurrent policies

For policies that take a fixed-length observation history as input (e.g. HIM-Loco style), use a group-level `history_steps` and optionally `interleaved`:

```json
"obs_config": {
  "obs_history": {
    "history_steps": 6,
    "interleaved": true,
    "components": [
      { "name": "SimpleVelocityCommand", "scale": [2.0, 2.0, 0.25] },
      { "name": "BaseAngularVelocity", "scale": 0.25 },
      { "name": "ProjectedGravity", "joint_name": "floating_base_joint" },
      { "name": "JointPositions", "joint_names": "isaac", "subtract_default": true },
      { "name": "JointVelocities", "joint_names": "isaac", "scale": 0.05 },
      { "name": "PreviousActions" }
    ]
  }
}
```

When `interleaved: true`, the history buffer is assembled by interleaving timesteps across all components (timestep 0 of all components, then timestep 1 of all components, …) rather than concatenating all history of each component in sequence.

## Minimal example

A minimal config for a simple locomotion policy with no optional fields:

```json
{
  "policy_joint_names": ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf",
                         "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf"],
  "default_joint_pos": [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
  "action_scale": 0.25,
  "stiffness": 40.0,
  "damping": 1.0,
  "control_type": "joint_position",
  "obs_config": {
    "obs": [
      { "name": "BaseAngularVelocity" },
      { "name": "ProjectedGravityB" },
      { "name": "JointPositions", "joint_names": "isaac", "subtract_default": true },
      { "name": "JointVelocities", "joint_names": "isaac" },
      { "name": "PreviousActions" },
      { "name": "SimpleVelocityCommand" }
    ]
  },
  "onnx": {
    "meta": {
      "in_keys": ["obs"],
      "out_keys": ["action"]
    }
  }
}
```
