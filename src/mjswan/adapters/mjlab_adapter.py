"""Adapter for converting mjlab types to mjswan internal representations.

mjlab is a **soft dependency** — this module never fails at import time.
When mjlab is not installed the ``adapt_*`` functions simply return their
inputs unchanged (they are assumed to already be mjswan types).

The adapter detects mjlab types by checking the module path of the class
(``type(obj).__module__``) rather than ``isinstance``, so mjlab does not
need to be importable for mjswan to function.

Mapping strategy
----------------
* **Observation functions**: mjlab callables (e.g.
  ``mjlab.envs.mdp.observations.base_lin_vel``) are matched by
  ``__name__`` to the corresponding :class:`ObsFunc` sentinel.
* **Termination functions**: same approach via ``__name__`` →
  :class:`TermFunc`.
* **Action configs**: mjlab ``ActionTermCfg`` subclasses are converted
  field-by-field to their mjswan equivalents.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

from ..envs.mdp import observations as _obs_sentinels
from ..envs.mdp import terminations as _term_sentinels
from ..envs.mdp.actions.actions import (
    ActionTermCfg as MjswanActionTermCfg,
    JointEffortActionCfg as MjswanJointEffortActionCfg,
    JointPositionActionCfg as MjswanJointPositionActionCfg,
)
from ..envs.mdp.observations import ObsFunc
from ..envs.mdp.terminations import TermFunc
from ..managers.observation_manager import (
    ObservationGroupCfg as MjswanObservationGroupCfg,
    ObservationTermCfg as MjswanObservationTermCfg,
)
from ..managers.termination_manager import (
    TerminationTermCfg as MjswanTerminationTermCfg,
)

# ---------------------------------------------------------------------------
# Function-name → sentinel registries
# ---------------------------------------------------------------------------

# Build mapping from function name to ObsFunc sentinel.
# Every module-level ObsFunc in mjswan.envs.mdp.observations is registered.
_OBS_FUNC_REGISTRY: dict[str, ObsFunc] = {
    name: obj for name, obj in vars(_obs_sentinels).items() if isinstance(obj, ObsFunc)
}

# Build mapping from function name to TermFunc sentinel.
_TERM_FUNC_REGISTRY: dict[str, TermFunc] = {
    name: obj
    for name, obj in vars(_term_sentinels).items()
    if isinstance(obj, TermFunc)
}


def _is_from_mjlab(obj: Any) -> bool:
    """Check whether *obj*'s class originates from the ``mjlab`` package."""
    module = getattr(type(obj), "__module__", "") or ""
    return module.startswith("mjlab")


def _is_mjlab_callable(func: Any) -> bool:
    """Check whether *func* is a callable originating from ``mjlab``."""
    module = getattr(func, "__module__", "") or ""
    return module.startswith("mjlab")


# ---------------------------------------------------------------------------
# Observation adaptation
# ---------------------------------------------------------------------------


def _adapt_obs_func(func: Any) -> ObsFunc:
    """Convert an mjlab observation callable to an mjswan ``ObsFunc`` sentinel.

    Raises ``ValueError`` if no mapping exists.
    """
    name = getattr(func, "__name__", None)
    if name and name in _OBS_FUNC_REGISTRY:
        return _OBS_FUNC_REGISTRY[name]
    raise ValueError(
        f"No mjswan mapping for mjlab observation function '{name}'. "
        f"Known mappings: {sorted(_OBS_FUNC_REGISTRY.keys())}"
    )


def _adapt_obs_term(term: Any) -> MjswanObservationTermCfg:
    """Convert a single mjlab ``ObservationTermCfg`` to mjswan."""
    obs_func = _adapt_obs_func(term.func)
    params: dict[str, Any] = dict(getattr(term, "params", None) or {})

    scale = getattr(term, "scale", None)
    clip = getattr(term, "clip", None)
    history_length = getattr(term, "history_length", 0) or 0

    return MjswanObservationTermCfg(
        func=obs_func,
        params=params,
        scale=scale,
        clip=clip,
        history_length=history_length,
    )


def _adapt_obs_group(group: Any) -> MjswanObservationGroupCfg:
    """Convert a single mjlab ``ObservationGroupCfg`` to mjswan."""
    terms: dict[str, MjswanObservationTermCfg] = {}
    raw_terms = getattr(group, "terms", None) or {}

    # mjlab stores terms as a dict[str, ObservationTermCfg]
    if isinstance(raw_terms, dict):
        for term_name, term_cfg in raw_terms.items():
            terms[term_name] = _adapt_obs_term(term_cfg)

    enable_corruption = getattr(group, "enable_corruption", False)
    concatenate_terms = getattr(group, "concatenate_terms", True)
    history_length = getattr(group, "history_length", None)

    return MjswanObservationGroupCfg(
        terms=terms,
        concatenate_terms=concatenate_terms,
        enable_corruption=enable_corruption,
        history_length=history_length,
    )


def adapt_observations(
    observations: dict[str, Any] | None,
) -> dict[str, MjswanObservationGroupCfg] | None:
    """Adapt observation groups, converting mjlab types if detected.

    If the values are already ``mjswan.ObservationGroupCfg`` instances they
    are returned as-is.  mjlab ``ObservationGroupCfg`` instances are
    converted transparently.

    Args:
        observations: Dict mapping group names to group configs (mjswan or
            mjlab), or ``None``.

    Returns:
        Dict of mjswan ``ObservationGroupCfg``, or ``None``.
    """
    if observations is None:
        return None

    result: dict[str, MjswanObservationGroupCfg] = {}
    for key, group in observations.items():
        if isinstance(group, MjswanObservationGroupCfg):
            result[key] = group
        elif _is_from_mjlab(group):
            result[key] = _adapt_obs_group(group)
        else:
            # Assume it's already mjswan-compatible
            result[key] = group
    return result


# ---------------------------------------------------------------------------
# Termination adaptation
# ---------------------------------------------------------------------------


def _adapt_term_func(func: Any) -> TermFunc:
    """Convert an mjlab termination callable to an mjswan ``TermFunc`` sentinel."""
    name = getattr(func, "__name__", None)
    if name and name in _TERM_FUNC_REGISTRY:
        return _TERM_FUNC_REGISTRY[name]
    raise ValueError(
        f"No mjswan mapping for mjlab termination function '{name}'. "
        f"Known mappings: {sorted(_TERM_FUNC_REGISTRY.keys())}"
    )


def _adapt_term_cfg(term: Any) -> MjswanTerminationTermCfg:
    """Convert a single mjlab ``TerminationTermCfg`` to mjswan."""
    term_func = _adapt_term_func(term.func)
    params: dict[str, Any] = dict(getattr(term, "params", None) or {})
    time_out: bool = getattr(term, "time_out", False)

    return MjswanTerminationTermCfg(
        func=term_func,
        params=params,
        time_out=time_out,
    )


def adapt_terminations(
    terminations: dict[str, Any] | None,
) -> dict[str, MjswanTerminationTermCfg] | None:
    """Adapt termination configs, converting mjlab types if detected.

    Args:
        terminations: Dict mapping term names to term configs (mjswan or
            mjlab), or ``None``.

    Returns:
        Dict of mjswan ``TerminationTermCfg``, or ``None``.
    """
    if terminations is None:
        return None

    result: dict[str, MjswanTerminationTermCfg] = {}
    for key, term in terminations.items():
        if isinstance(term, MjswanTerminationTermCfg):
            result[key] = term
        elif _is_from_mjlab(term):
            result[key] = _adapt_term_cfg(term)
        else:
            result[key] = term
    return result


# ---------------------------------------------------------------------------
# Action adaptation
# ---------------------------------------------------------------------------

# mjlab class name → mjswan adapter function
_ACTION_CLASS_MAP: dict[str, str] = {
    "JointPositionActionCfg": "joint_position",
    "JointEffortActionCfg": "joint_effort",
    "JointVelocityActionCfg": "joint_velocity",
}


def _adapt_action_cfg(term: Any) -> MjswanActionTermCfg:
    """Convert a single mjlab ``ActionTermCfg`` to mjswan."""
    class_name = type(term).__name__

    # Extract common fields
    scale = getattr(term, "scale", 1.0)
    offset = getattr(term, "offset", 0.0)
    entity_name = getattr(term, "entity_name", "robot")
    clip = getattr(term, "clip", None)
    actuator_names = getattr(term, "actuator_names", (".*",))

    if class_name == "JointPositionActionCfg":
        use_default_offset = getattr(term, "use_default_offset", True)
        stiffness = getattr(term, "stiffness", None)
        damping = getattr(term, "damping", None)
        return MjswanJointPositionActionCfg(
            entity_name=entity_name,
            clip=clip,
            actuator_names=actuator_names,
            scale=scale,
            offset=offset,
            use_default_offset=use_default_offset,
            stiffness=stiffness,
            damping=damping,
        )
    elif class_name == "JointEffortActionCfg":
        stiffness = getattr(term, "stiffness", None)
        damping = getattr(term, "damping", None)
        return MjswanJointEffortActionCfg(
            entity_name=entity_name,
            clip=clip,
            actuator_names=actuator_names,
            scale=scale,
            offset=offset,
            stiffness=stiffness,
            damping=damping,
        )
    else:
        warnings.warn(
            f"mjlab action type '{class_name}' has no mjswan equivalent. "
            f"It will be skipped at build time.",
            category=RuntimeWarning,
            stacklevel=3,
        )
        # Return a stub that raises at serialization time
        from ..envs.mdp.actions.actions import BaseActionCfg

        return BaseActionCfg(
            entity_name=entity_name,
            clip=clip,
            actuator_names=actuator_names,
            scale=scale,
            offset=offset,
            unsupported_reason=(
                f"mjlab action type '{class_name}' is not supported in mjswan."
            ),
        )


def adapt_actions(
    actions: Mapping[str, Any] | None,
) -> Mapping[str, MjswanActionTermCfg] | None:
    """Adapt action configs, converting mjlab types if detected.

    Args:
        actions: Dict mapping term names to action configs (mjswan or
            mjlab), or ``None``.

    Returns:
        Dict of mjswan ``ActionTermCfg``, or ``None``.
    """
    if actions is None:
        return None

    result: dict[str, MjswanActionTermCfg] = {}
    for key, term in actions.items():
        if isinstance(term, MjswanActionTermCfg):
            result[key] = term
        elif _is_from_mjlab(term):
            result[key] = _adapt_action_cfg(term)
        else:
            result[key] = term
    return result


__all__ = ["adapt_observations", "adapt_actions", "adapt_terminations"]
