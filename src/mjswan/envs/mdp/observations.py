"""Built-in observation function sentinels for mjswan.

Each object mirrors a function from ``mjlab.envs.mdp.observations``.
In mjswan these are **not** called at runtime — they are sentinel
objects that carry metadata mapping to the TypeScript observation class
used by the browser-side ``PolicyRunner``.

Usage (identical to mjlab)::

    from mjswan.envs.mdp import observations as obs_fns
    from mjswan.managers.observation_manager import ObservationTermCfg

    ObservationTermCfg(func=obs_fns.base_lin_vel)
    ObservationTermCfg(func=obs_fns.joint_pos_rel, scale=0.5)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObsFunc:
    """Sentinel representing a JS-side observation implementation.

    Attributes:
        ts_name: The TypeScript observation class name in the
            ``Observations`` registry (e.g. ``"BaseLinearVelocity"``).
        defaults: Default parameters merged into the JSON config entry.
            These map mjlab semantics to the existing TS class API.
    """

    ts_name: str
    defaults: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Root state
# ---------------------------------------------------------------------------

base_lin_vel = ObsFunc("BaseLinearVelocity", {"world_frame": False})
"""Linear velocity of the robot base in the base frame.

mjlab: ``asset.data.root_link_lin_vel_b``
"""

base_ang_vel = ObsFunc("BaseAngularVelocity")
"""Angular velocity of the robot base in the base frame.

mjlab: ``asset.data.root_link_ang_vel_b``
"""

projected_gravity = ObsFunc("ProjectedGravityB")
"""Gravity vector projected into the base frame.

mjlab: ``asset.data.projected_gravity_b``
"""

# ---------------------------------------------------------------------------
# Joint state
# ---------------------------------------------------------------------------

joint_pos_rel = ObsFunc("JointPos", {"subtract_default": True})
"""Joint positions relative to the default pose.

mjlab: ``asset.data.joint_pos - asset.data.default_joint_pos``
"""

joint_vel_rel = ObsFunc("JointVelocities")
"""Joint velocities relative to the default velocities.

mjlab: ``asset.data.joint_vel - asset.data.default_joint_vel``
"""

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

last_action = ObsFunc("PrevActions", {"history_steps": 1})
"""The most recent action tensor.

mjlab: ``env.action_manager.action``
"""

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

generated_commands = ObsFunc("GeneratedCommands")
"""Current command tensor from a named command term.

Requires ``params={"command_name": "<name>"}``.

mjlab: ``env.command_manager.get_command(command_name)``
"""


__all__ = [
    "ObsFunc",
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    "generated_commands",
]
