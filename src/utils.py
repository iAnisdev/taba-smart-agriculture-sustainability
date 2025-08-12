from __future__ import annotations
import json, os, time
from pathlib import Path
from typing import Dict, Any

import torch

def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(obj: Dict[str, Any], path: str):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def model_size_mb_from_state_dict(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.t1 = time.time()
        self.elapsed = self.t1 - self.t0
