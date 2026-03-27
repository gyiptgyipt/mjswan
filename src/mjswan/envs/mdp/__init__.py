"""MDP components for mjswan.

Mirrors ``mjlab.envs.mdp``.  Re-exports the ``observations`` module so
that the following mjlab import pattern translates directly::

    # mjlab
    from mjlab.envs.mdp import observations as obs_fns

    # mjswan (identical)
    from mjswan.envs.mdp import observations as obs_fns
"""

from . import observations

__all__ = ["observations"]
