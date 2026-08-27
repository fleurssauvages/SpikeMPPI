from __future__ import annotations

from pathlib import Path

from .base import JointPolicy
from .default import ClassicNeutralPolicy
from .external import load_callable_policy


def _auto_checkpoint(robot_name: str | None) -> Path | None:
    if not robot_name:
        return None
    robot = str(robot_name).strip().lower().replace("-", "_")
    candidates = [
        Path(__file__).resolve().parent / "checkpoints" / f"{robot}_rapid",
        Path.cwd() / "racing" / "policies" / "checkpoints" / f"{robot}_rapid",
    ]
    for path in candidates:
        if (path / "metadata.json").exists() and (path / "params.pkl").exists():
            return path
    return None


def make_policy(
    spec: str | None,
    *,
    race_speed: float | None = None,
    robot_name: str | None = None,
) -> JointPolicy:
    text = "default" if spec is None else str(spec).strip()
    key = text.lower()

    if key in {"auto", "rapid", "fast"}:
        checkpoint = _auto_checkpoint(robot_name)
        if checkpoint is None:
            raise FileNotFoundError(
                f"No trained rapid-locomotion checkpoint found for {robot_name!r}. "
                f"Expected racing/policies/checkpoints/{robot_name}_rapid."
            )
        from .brax_velocity import BraxVelocityPolicy

        return BraxVelocityPolicy(checkpoint, race_speed=race_speed)

    if key in {"", "default", "neutral", "zero"}:
        return ClassicNeutralPolicy()

    if text.startswith("velocity:"):
        text = text.split(":", 1)[1]
    path = Path(text).expanduser()
    if path.exists() and path.is_dir():
        from .brax_velocity import BraxVelocityPolicy

        return BraxVelocityPolicy(path, race_speed=race_speed)

    return load_callable_policy(text)
