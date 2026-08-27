from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class SpatialPrior(ABC):
    """Trajectory prior indexed by track progress s, never by wall-clock time."""

    @abstractmethod
    def sample(self, track, s: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """Return mean XY and 2x2 covariance at progress values ``s``."""
        raise NotImplementedError
