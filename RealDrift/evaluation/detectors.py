from __future__ import annotations

from dataclasses import dataclass, field

REGISTRY = {}


@dataclass
class DetectionResult:
    timestamps: list
    window_seconds: float | None = None
    info: dict = field(default_factory=dict)


def register(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def get(name):
    if name not in REGISTRY:
        raise KeyError(f"detector {name!r} is not registered, known detectors {sorted(REGISTRY)}. "
                       f"Add its module to the detectors section of configs/detectors.yaml.")
    return REGISTRY[name]
