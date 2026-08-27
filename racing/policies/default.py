from __future__ import annotations

import numpy as np
from .base import JointPolicy


class ClassicNeutralPolicy(JointPolicy):
    """Neutral nominal for the classic MuJoCo robots.

    This is intentionally not a learned locomotion policy. It returns the
    model's native neutral actuator vector and exists so standard MPPI can be
    tested without adding another policy dependency. A trained policy can be
    supplied later through ``module:function``.
    """

    name = 'classic_neutral'

    def action(self, robot, data, *, track, prior, current_s: float) -> np.ndarray:
        del data, track, prior, current_s
        return robot.default_ctrl.copy()
