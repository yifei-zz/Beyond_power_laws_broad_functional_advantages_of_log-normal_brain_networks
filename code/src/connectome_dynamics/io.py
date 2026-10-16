"""Serialization and filesystem helpers."""
from __future__ import annotations
import pickle
from pathlib import Path

def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def save_pickle(obj, path: Path) -> None:
    with ensure_parent(path).open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)
