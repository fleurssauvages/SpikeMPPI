from __future__ import annotations

from .base import ClassicRobot
from .classic import find_classic_model, list_classic_models


def _robot_variant(name: str) -> tuple[str, str, str, str]:
    """Return base model, actuator model, public name and display name."""
    key = str(name).strip().lower().replace("_", "-").replace(" ", "-")
    if key == "ant":
        return "ant", "motor", "ant", "Ant"
    if key == "ant-bio":
        return "ant", "muscle", "ant-bio", "Ant-Bio (antagonistic muscles)"
    raise KeyError("Unknown robot %r. Available: ant, ant-bio" % (name,))


def make_robot(
    name: str,
    *,
    extra_worldbody_xml: str = "",
    leg_length_scales: dict[str, float] | None = None,
) -> ClassicRobot:
    base, actuator_model, public_name, display_name = _robot_variant(name)
    return ClassicRobot(
        find_classic_model(base),
        extra_worldbody_xml=extra_worldbody_xml,
        leg_length_scales=leg_length_scales,
        actuator_model=actuator_model,
        variant_name=public_name,
        variant_display_name=display_name,
    )


def list_racing_candidates() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in ("ant", "ant-bio"):
        try:
            robot = make_robot(name)
            rows.append({
                "name": robot.name,
                "display_name": robot.display_name,
                "navigation": robot.navigation,
                "stadium": robot.supports_stadium,
                "nu": robot.nu,
                "nq": int(robot.model.nq),
                "nv": int(robot.model.nv),
                "na": int(robot.model.na),
                "actuator_model": robot.actuator_model,
                "root_body": robot.root_body_name,
                "xml": str(robot.xml_path),
            })
        except Exception as exc:
            rows.append({"name": name, "error": str(exc)})
    return rows


__all__ = ["make_robot", "list_racing_candidates"]
