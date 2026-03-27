"""Manager modules for mjswan.

Mirrors the ``mjlab.managers`` package layout so that mjlab import paths
translate directly::

    # mjlab
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

    # mjswan (identical API)
    from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
"""

from .observation_manager import ObservationGroupCfg, ObservationTermCfg

__all__ = ["ObservationGroupCfg", "ObservationTermCfg"]
