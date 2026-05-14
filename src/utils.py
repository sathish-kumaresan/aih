import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_yaml(path) -> SimpleNamespace:
    with open(path) as f:
        return _to_namespace(yaml.safe_load(f))


def dump_yaml(obj, path) -> None:
    def _plain(o):
        if isinstance(o, SimpleNamespace):
            return {k: _plain(v) for k, v in vars(o).items()}
        if isinstance(o, list):
            return [_plain(v) for v in o]
        if isinstance(o, Path):
            return str(o)
        return o
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(_plain(obj), f, sort_keys=False)


def run_dir(tier: int, protocol: str, seed: int, root: str = "results/runs") -> Path:
    p = Path(root) / f"tier{tier}" / protocol / f"seed{seed}"
    p.mkdir(parents=True, exist_ok=True)
    return p
