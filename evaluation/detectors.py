"""Detector interface.

A detector is a function  detect(xes_path: str, params: dict) -> DetectionResult  registered under
a name. Implement PELT, Version Clustering or any other method in its own module and register it
there, the runner imports the modules listed in configs/detectors.yaml.

    from evaluation.detectors import DetectionResult, register

    @register("pelt")
    def detect(xes_path, params):
        ...
        return DetectionResult(timestamps=[...], window_seconds=..., info={...})

window_seconds is the duration of one analysis window (the resolution of the detector). It is only
needed when the tolerance is given in windows.
"""
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
