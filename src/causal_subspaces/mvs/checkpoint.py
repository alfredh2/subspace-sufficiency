"""Checkpoints hold only the trainable basis, mu, optimizer/RNG state and metadata (never base weights)."""
from pathlib import Path

import torch


def save_checkpoint(path: Path, *, V, U, mu, step, optimizer=None, rng=None, meta=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(V=V.detach().cpu(), U=U.detach().cpu(), mu=mu.cpu(), step=step,
                    optimizer=optimizer.state_dict() if optimizer else None, rng=rng, meta=meta or {}), path)


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    cks = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return cks[-1] if cks else None
