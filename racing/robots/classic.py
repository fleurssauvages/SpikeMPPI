from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class ClassicModel:
    name: str
    xml_name: str
    display_name: str
    navigation: str  # 'xy' or 'x'
    root_body_hint: str = 'torso'


CLASSIC_MODELS: Dict[str, ClassicModel] = {
    'ant': ClassicModel('ant', 'ant.xml', 'Ant', 'xy'),
    'humanoid': ClassicModel('humanoid', 'humanoid.xml', 'Humanoid', 'xy'),
    'swimmer': ClassicModel('swimmer', 'swimmer.xml', 'Swimmer', 'xy'),
    'half_cheetah': ClassicModel('half_cheetah', 'half_cheetah.xml', 'HalfCheetah', 'x'),
    'hopper': ClassicModel('hopper', 'hopper.xml', 'Hopper', 'x'),
    'walker2d': ClassicModel('walker2d', 'walker2d.xml', 'Walker2d', 'x'),
}


def _asset_root() -> Path:
    try:
        import gymnasium
    except ImportError as exc:
        raise RuntimeError(
            'The classic MuJoCo robot XMLs are loaded from Gymnasium. Install them with:\n'
            '  pip install "gymnasium[mujoco]"'
        ) from exc
    root = Path(gymnasium.__file__).resolve().parent / 'envs' / 'mujoco' / 'assets'
    if not root.exists():
        raise FileNotFoundError(
            f'Gymnasium MuJoCo assets were not found at {root}. '
            'Install/repair with: pip install "gymnasium[mujoco]"'
        )
    return root


def list_classic_models() -> list[ClassicModel]:
    return list(CLASSIC_MODELS.values())


def find_classic_model(name: str) -> ClassicModel:
    key = str(name).strip().lower().replace('-', '_').replace(' ', '_')
    aliases = {
        'halfcheetah': 'half_cheetah',
        'cheetah': 'half_cheetah',
        'walker': 'walker2d',
        'walker_2d': 'walker2d',
    }
    key = aliases.get(key, key)
    if key not in CLASSIC_MODELS:
        raise KeyError(
            f'Unknown classic MuJoCo robot {name!r}. '
            f'Available: {", ".join(CLASSIC_MODELS)}'
        )
    return CLASSIC_MODELS[key]


def classic_xml_path(info: ClassicModel) -> Path:
    path = _asset_root() / info.xml_name
    if not path.exists():
        raise FileNotFoundError(f'Classic MuJoCo asset is missing: {path}')
    return path
