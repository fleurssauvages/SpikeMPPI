from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParameterScales:
    """Small set of environment/dynamics parameters adapted online."""

    friction: float = 1.0
    mass: float = 1.0
    motor: float = 1.0
    slope_deg: float = 0.0

    def clipped(
        self,
        *,
        friction_bounds=(0.15, 3.0),
        mass_bounds=(0.5, 1.8),
        motor_bounds=(0.5, 1.8),
        slope_bounds=(-20.0, 20.0),
    ) -> "ModelParameterScales":
        import numpy as np

        return ModelParameterScales(
            friction=float(np.clip(self.friction, *friction_bounds)),
            mass=float(np.clip(self.mass, *mass_bounds)),
            motor=float(np.clip(self.motor, *motor_bounds)),
            slope_deg=float(np.clip(self.slope_deg, *slope_bounds)),
        )


__all__ = ["ModelParameterScales"]
