"""YAML configs: an experiment file names model/data/hardware files; CLI `--set a.b=c` overrides."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    base = {}
    for key in ("model", "data", "hardware"):  # `defaults: {model: qwen3_8b, ...}`
        name = cfg.get("defaults", {}).get(key)
        if name:
            base = _merge(base, {key: yaml.safe_load((ROOT / "configs" / key / f"{name}.yaml").read_text())})
    cfg = _merge(base, {k: v for k, v in cfg.items() if k != "defaults"})
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(v)
    return cfg
