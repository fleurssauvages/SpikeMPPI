from __future__ import annotations

import importlib
from typing import Callable
import numpy as np

from .base import JointPolicy


class CallableJointPolicy(JointPolicy):
    """Wrap a user/pretrained policy callable.

    Callable signature::

        fn(robot, data, track, prior, current_s) -> ctrl[nu]

    This is the intended bridge for a trained MuJoCo Playground policy or any
    robot-specific locomotion controller without changing the MPPI code.
    """

    def __init__(self, fn: Callable, name: str | None = None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "callable_policy")

    def action(self, robot, data, *, track, prior, current_s: float) -> np.ndarray:
        out = self.fn(robot, data, track, prior, float(current_s))
        out = np.asarray(out, dtype=np.float64).reshape(-1)
        if out.shape != (robot.nu,):
            raise ValueError(f"Policy {self.name!r} returned {out.shape}; expected ({robot.nu},)")
        return robot.clip_ctrl(out)


def load_callable_policy(spec: str) -> CallableJointPolicy:
    """Load ``module:function`` from the active Python environment."""
    if ":" not in spec:
        raise ValueError("External policy must be written as module:function")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name)
    return CallableJointPolicy(fn, name=spec)
