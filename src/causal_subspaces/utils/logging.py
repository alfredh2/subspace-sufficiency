"""Run directories and machine-readable logging (JSONL / JSON / YAML). No remote services required."""
import json
import time
from pathlib import Path

import yaml


class RunDir:
    def __init__(self, results_root: str | Path, run_name: str, exist_ok: bool = False):
        self.path = Path(results_root) / run_name
        self.path.mkdir(parents=True, exist_ok=exist_ok)
        (self.path / "checkpoints").mkdir(exist_ok=True)

    def write_config(self, cfg: dict) -> None:
        (self.path / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    def write_json(self, name: str, obj) -> None:
        (self.path / name).write_text(json.dumps(obj, indent=2, default=str))

    def jsonl(self, name: str, mode: str = "a") -> "JsonlWriter":
        return JsonlWriter(self.path / name, mode)


class JsonlWriter:
    def __init__(self, path: Path, mode: str = "a"):
        self.f = open(path, mode, buffering=1)
        self.t0 = time.time()

    def write(self, **row) -> None:
        row.setdefault("elapsed_s", round(time.time() - self.t0, 2))
        self.f.write(json.dumps(row, default=float) + "\n")

    def close(self) -> None:
        self.f.close()
