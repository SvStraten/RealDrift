"""PELT drift detector (Adams et al., Explainable Concept Drift in Process Mining).

Uses the feature series and PELT step of the evaluation pipeline in evaluation/pelt_impl (copied
from drift_generation_evaluation_pipeline). The original reports change points per feature. For a
single set of detections per log the change points of all features are pooled, change points of
different features that lie within `merge_windows` windows of each other form one detection, and a
detection is kept when at least `min_features` features support it. The detection time is the
start of the median window of its group.

params
    window_days     window size in days (7 in the original)
    penalty         PELT penalty on the max-normalised series (0.1 in the original)
    min_features    number of features that must agree (1 keeps the pooled change points)
    merge_windows   change points at most this many windows apart count as one detection
    trim, auto_trim_frac   edge handling of the original (auto_trim_frac 0.05)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from evaluation.detectors import DetectionResult, register
from evaluation.pelt_impl.drift_detection import (AUTO_TRIM_FRAC_DEFAULT, PELT_PENALTY_DEFAULT,
                                                  WINDOW_DAYS_DEFAULT, build_time_series,
                                                  detect_change_points, sparse_edge_bounds)
from evaluation.pelt_impl.xes_loading import data_attribute_columns, load_xes


def pool_change_points(change_points: dict, merge_windows: int, min_features: int) -> list:
    """Group change points of all features and return one window index per supported group."""
    events = sorted((cp, feat) for feat, cps in change_points.items() for cp in cps)
    groups, current = [], []
    for cp, feat in events:
        if current and cp - current[-1][0] > merge_windows:
            groups.append(current)
            current = []
        current.append((cp, feat))
    if current:
        groups.append(current)
    kept = []
    for g in groups:
        if len({f for _, f in g}) >= min_features:
            kept.append(int(np.median([cp for cp, _ in g])))
    return kept


_SERIES_CACHE = {}


def _series(xes_path, window_days, trim, auto_trim):
    """Feature series per (log, window, trim), shared by all penalties of a grid search."""
    key = (xes_path, os.path.getmtime(xes_path), window_days, trim, auto_trim)
    if key not in _SERIES_CACHE:
        if len(_SERIES_CACHE) > 8:
            _SERIES_CACHE.clear()
        df, info = load_xes(xes_path)
        numeric = [a for a in data_attribute_columns(df, info) if pd.api.types.is_numeric_dtype(df[a])]
        series, time_index, counts, _ = build_time_series(df, numeric, info["has_start"], window_days)
        lo, hi = 0, len(time_index)
        if auto_trim > 0:
            lo, hi = sparse_edge_bounds(counts, auto_trim)
        lo, hi = lo + trim, hi - trim
        if hi - lo >= 5:
            series = {k: v[lo:hi] for k, v in series.items()}
            time_index = time_index[lo:hi]
        _SERIES_CACHE[key] = (series, time_index)
    return _SERIES_CACHE[key]


@register("pelt")
def detect(xes_path, params):
    window_days = int(params.get("window_days", WINDOW_DAYS_DEFAULT))
    penalty = float(params.get("penalty", PELT_PENALTY_DEFAULT))
    min_features = int(params.get("min_features", 1))
    merge_windows = int(params.get("merge_windows", 1))
    trim = int(params.get("trim", 0))
    auto_trim = float(params.get("auto_trim_frac", AUTO_TRIM_FRAC_DEFAULT))

    series, time_index = _series(xes_path, window_days, trim, auto_trim)
    cps = detect_change_points(series, penalty)
    windows = pool_change_points(cps, merge_windows, min_features)
    stamps = [pd.Timestamp(time_index[w]) for w in windows if w < len(time_index)]
    return DetectionResult(
        timestamps=stamps, window_seconds=window_days * 86400.0,
        info={"n_windows": len(time_index), "n_features": len(series),
              "n_raw_change_points": int(sum(len(v) for v in cps.values()))})
