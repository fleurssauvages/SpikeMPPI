from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class JointPolicy(ABC):
    """Nominal joint-control policy used only to seed the MPPI/iLQR proposal.

    The sampled MPPI rollouts do *not* call this policy. MPPI acts directly on
    the robot's MuJoCo actuator vector ``data.ctrl``.
    """

    name: str = "policy"

    def reset(self, robot, data) -> None:
        del robot, data

    @abstractmethod
    def action(self, robot, data, *, track, prior, current_s: float) -> np.ndarray:
        """Return one nominal actuator vector with shape ``(model.nu,)``."""
        raise NotImplementedError
