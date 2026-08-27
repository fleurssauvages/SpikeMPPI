from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import numpy as np

from .model_params import ModelParameterScales


@dataclass
class SystemIDConfig:
    history: int = 12
    update_interval: int = 8
    smoothing: float = 0.35
    friction_step: float = 0.20
    mass_step: float = 0.08
    motor_step: float = 0.08
    slope_step_deg: float = 1.5
    estimate_friction: bool = True
    estimate_mass: bool = True
    estimate_motor: bool = True
    estimate_slope: bool = False
    root_position_weight: float = 8.0
    root_height_weight: float = 3.0
    velocity_weight: float = 1.0
    joint_position_weight: float = 0.15


@dataclass
class Transition:
    before: object
    ctrl: np.ndarray
    after: object
    substeps: int


class OnlineSystemIdentifier:
    """Tiny derivative-free online identifier for the MPPI MuJoCo model.

    The physical plant and planning model are separate ``ClassicRobot`` objects.
    Recent real transitions are replayed through the planning model under nearby
    parameter hypotheses.  A coordinate search updates friction, mass, actuator
    strength, and optionally effective slope.
    """

    def __init__(self, planner_robot, config: SystemIDConfig | None = None) -> None:
        self.robot = planner_robot
        self.cfg = config or SystemIDConfig()
        self.history: deque[Transition] = deque(maxlen=max(2, int(self.cfg.history)))
        self.estimate = ModelParameterScales()
        self.update_count = 0
        self.last_loss = math.inf
        self.history_estimates: list[ModelParameterScales] = [self.estimate]
        self.robot.apply_model_parameters(self.estimate)

    def observe(self, before, ctrl: np.ndarray, after, *, substeps: int) -> None:
        self.history.append(
            Transition(before, np.asarray(ctrl, dtype=np.float64).copy(), after, int(substeps))
        )

    def should_update(self, step: int) -> bool:
        return (
            len(self.history) >= max(3, min(self.cfg.history, 5))
            and (int(step) + 1) % max(1, int(self.cfg.update_interval)) == 0
        )

    def _transition_loss(self, transition: Transition) -> float:
        d = self.robot.new_data(transition.before)
        self.robot.step_control(transition.ctrl, substeps=transition.substeps, data=d)
        pred = self.robot.snapshot(d)
        obs = transition.after

        # qpos[0:3] is the free-root world position for Ant/Humanoid.
        root_xy = float(np.mean((pred.qpos[:2] - obs.qpos[:2]) ** 2))
        root_z = float((pred.qpos[2] - obs.qpos[2]) ** 2)
        vel_scale = np.maximum(1.0, np.abs(obs.qvel))
        qvel = float(np.mean(((pred.qvel - obs.qvel) / vel_scale) ** 2))
        if len(obs.qpos) > 7:
            qpos_joint = float(np.mean((pred.qpos[7:] - obs.qpos[7:]) ** 2))
        else:
            qpos_joint = 0.0
        return (
            self.cfg.root_position_weight * root_xy
            + self.cfg.root_height_weight * root_z
            + self.cfg.velocity_weight * qvel
            + self.cfg.joint_position_weight * qpos_joint
        )

    def _loss(self, params: ModelParameterScales) -> float:
        self.robot.apply_model_parameters(params)
        if not self.history:
            return math.inf
        values = [self._transition_loss(t) for t in self.history]
        # Recent transitions matter slightly more.
        weights = np.linspace(0.5, 1.0, len(values), dtype=np.float64)
        return float(np.average(np.asarray(values), weights=weights))

    @staticmethod
    def _replace(p: ModelParameterScales, name: str, value: float) -> ModelParameterScales:
        kwargs = dict(
            friction=p.friction,
            mass=p.mass,
            motor=p.motor,
            slope_deg=p.slope_deg,
        )
        kwargs[name] = float(value)
        return ModelParameterScales(**kwargs).clipped()

    def update(self) -> ModelParameterScales:
        current = self.estimate
        best = current
        best_loss = self._loss(current)

        axes: list[tuple[str, float]] = []
        if self.cfg.estimate_friction:
            axes.append(("friction", float(self.cfg.friction_step)))
        if self.cfg.estimate_mass:
            axes.append(("mass", float(self.cfg.mass_step)))
        if self.cfg.estimate_motor:
            axes.append(("motor", float(self.cfg.motor_step)))
        if self.cfg.estimate_slope:
            axes.append(("slope_deg", float(self.cfg.slope_step_deg)))

        # One coordinate-descent sweep.  This is deliberately small so it can
        # run online between MPPI updates without turning system ID into the
        # dominant computation.
        for name, step in axes:
            base_value = float(getattr(best, name))
            candidates = [
                self._replace(best, name, base_value - step),
                best,
                self._replace(best, name, base_value + step),
            ]
            local_best = best
            local_loss = best_loss
            for candidate in candidates:
                loss = self._loss(candidate)
                if loss < local_loss:
                    local_loss = loss
                    local_best = candidate
            best, best_loss = local_best, local_loss

        alpha = float(np.clip(self.cfg.smoothing, 0.0, 1.0))
        smoothed = ModelParameterScales(
            friction=(1.0 - alpha) * current.friction + alpha * best.friction,
            mass=(1.0 - alpha) * current.mass + alpha * best.mass,
            motor=(1.0 - alpha) * current.motor + alpha * best.motor,
            slope_deg=(1.0 - alpha) * current.slope_deg + alpha * best.slope_deg,
        ).clipped()
        self.estimate = smoothed
        self.robot.apply_model_parameters(smoothed)
        self.last_loss = float(best_loss)
        self.update_count += 1
        self.history_estimates.append(smoothed)
        return smoothed


__all__ = ["SystemIDConfig", "OnlineSystemIdentifier"]
