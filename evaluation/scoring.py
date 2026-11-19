"""Scoring of detected drift points against the composed ground truth.

Ground truth. Every transition has an interval [start_ts, end_ts]. It is a single instant for a
sudden drift and the span in which both tasks admit cases for a gradual drift.

Matching. A detection matches a drift if it lies within `tolerance` of the interval, that is
inside it or at most `tolerance` outside it. The matching is one-to-one and maximizes the number
of matched pairs, ties are broken by the smallest total distance (optimal assignment, not a greedy
pass). Unmatched detections are false positives, unmatched drifts false negatives.

Conventions. Precision is 0 when nothing is detected, recall is 0 when there is no ground truth
drift, and F1 is 0 when precision and recall are both 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass
class Score:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    n_gt: int
    n_det: int
    distances_days: list = field(default_factory=list)


def load_ground_truth(path_or_frame) -> pd.DataFrame:
    gt = pd.read_csv(path_or_frame) if isinstance(path_or_frame, str) else path_or_frame.copy()
    for c in ("start_ts", "end_ts"):
        gt[c] = pd.to_datetime(gt[c], utc=True, format="ISO8601")
    return gt.sort_values("start_ts").reset_index(drop=True)


def resolve_tolerance(spec, window_seconds=None, span_seconds=None) -> pd.Timedelta:
    """spec is {unit: days, value: 7}, {unit: windows, value: 1} or {unit: span_fraction, value: 0.02}.
    'windows' is a multiple of the resolution of the detector (the duration of one analysis window),
    which the detector reports. 'span_fraction' is a share of the time span of the whole stream, so
    that logs with very different time scales get a comparable tolerance."""
    unit, value = spec["unit"], float(spec["value"])
    if unit == "days":
        return pd.Timedelta(days=value)
    if unit == "hours":
        return pd.Timedelta(hours=value)
    if unit == "windows":
        if not window_seconds:
            raise ValueError("tolerance in windows needs the detector to report window_seconds")
        return pd.Timedelta(seconds=value * float(window_seconds))
    if unit == "span_fraction":
        if not span_seconds:
            raise ValueError("tolerance as span_fraction needs the time span of the stream")
        return pd.Timedelta(seconds=value * float(span_seconds))
    raise ValueError(f"unknown tolerance unit {unit!r}")


def _distance(d, start, end):
    if start <= d <= end:
        return pd.Timedelta(0)
    return min(abs(d - start), abs(d - end))


def score_detections(detections, gt: pd.DataFrame, tolerance: pd.Timedelta) -> Score:
    dets = sorted(pd.Timestamp(d) if pd.Timestamp(d).tzinfo else pd.Timestamp(d).tz_localize("UTC")
                  for d in detections)
    n_gt, n_det = len(gt), len(dets)
    matched, dists = 0, []
    if n_gt and n_det:
        dist = np.array([[_distance(d, r.start_ts, r.end_ts).total_seconds() for d in dets]
                         for r in gt.itertuples()])
        feasible = dist <= tolerance.total_seconds()
        big = 1e12   # cardinality first, then total distance
        cost = np.where(feasible, dist - big, 0.0)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if feasible[r, c]:
                matched += 1
                dists.append(dist[r, c] / 86400)
    tp, fp, fn = matched, n_det - matched, n_gt - matched
    precision = tp / n_det if n_det else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return Score(tp, fp, fn, precision, recall, f1, n_gt, n_det, dists)
