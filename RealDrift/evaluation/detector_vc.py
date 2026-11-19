"""Version Clustering drift detector (Abarca Zuniga, Yeshchenko and van der Aa, CAiSE 2026)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from pm4py.objects.log.obj import EventLog

from evaluation.detectors import DetectionResult, register
from evaluation.vc_impl import preprocessing_methods as ppm
from evaluation.vc_impl import utilities as utils
from evaluation.vc_impl.clustering import cluster_hdbscan, find_transition_steps

_CACHE = {}


def _sorted_by_start(log):
    traces = [t for t in log if len(t) > 0]
    traces.sort(key=lambda t: min(e["time:timestamp"] for e in t))
    return EventLog(traces, attributes=log.attributes)


def _windows_and_clusters(xes_path, n_windows, min_cluster_size, threshold):
    key = (xes_path, os.path.getmtime(xes_path), n_windows, min_cluster_size, threshold)
    if key not in _CACHE:
        if len(_CACHE) > 16:
            _CACHE.clear()
        log = utils.filter_complete_events(utils.import_event_log(os.path.dirname(xes_path) or ".",
                                                                  os.path.basename(xes_path)))
        log = _sorted_by_start(log)
        if len(log) < 2 * n_windows:
            raise ValueError(f"{len(log)} traces are too few for {n_windows} windows")
        dfg, _borders, window_info, _date_info = ppm.log_to_windowed_dfg_count(log, n_windows)
        similarity = ppm.similarity_calculation(np.array(dfg))
        probabilities, labels = cluster_hdbscan(similarity, min_cluster_size=min_cluster_size,
                                                threshold=threshold)
        first = min(t[0]["time:timestamp"] for t in log)
        last = max(t[-1]["time:timestamp"] for t in log)
        _CACHE[key] = (probabilities, labels, window_info, (last - first).total_seconds())
    return _CACHE[key]


@register("vc")
def detect(xes_path, params):
    n_windows = int(params.get("n_windows", 75))
    min_cluster_size = int(params.get("min_cluster_size", 5))
    pen = float(params.get("pen", 1.0))
    threshold = float(params.get("threshold", 0.5))   # only used for labelling, not for detection

    probabilities, labels, window_info, span = _windows_and_clusters(
        xes_path, n_windows, min_cluster_size, threshold)
    change_points = find_transition_steps(probabilities, pen, n_windows)
    stamps = [pd.Timestamp(window_info[int(c) + 1][1], tz="UTC") for c in change_points
              if 0 < int(c) < n_windows]
    return DetectionResult(timestamps=stamps, window_seconds=span / n_windows,
                           info={"n_versions": int(len(set(labels)) - (1 if -1 in labels else 0)),
                                 "change_windows": [int(c) for c in change_points]})
