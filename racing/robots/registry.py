from __future__ import annotations

from .base import ClassicRobot
from .classic import find_classic_model, list_classic_models


def make_robot(name: str) -> ClassicRobot:
    return ClassicRobot(find_classic_model(name))


def list_racing_candidates() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for info in list_classic_models():
        try:
            robot = ClassicRobot(info)
            rows.append({
                'name': info.name,
                'display_name': info.display_name,
                'navigation': info.navigation,
                'stadium': robot.supports_stadium,
                'nu': robot.nu,
                'nq': int(robot.model.nq),
                'nv': int(robot.model.nv),
                'root_body': robot.root_body_name,
                'xml': str(robot.xml_path),
            })
        except Exception as exc:
            rows.append({'name': info.name, 'display_name': info.display_name, 'error': str(exc)})
    return rows


__all__ = ['make_robot', 'list_racing_candidates']
