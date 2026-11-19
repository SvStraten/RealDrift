""""
Implementation of the framework of Adams et al., "Explainable Concept Drift in
Process Mining" (Information Systems, 2023).
"""
from datetime import timedelta

import numpy as np
import pandas as pd
import ruptures as rpt
from statsmodels.tools import sm_exceptions
from statsmodels.tsa.stattools import grangercausalitytests

from .xes_loading import ACTIVITY, CASE, END, RESOURCE, START

WINDOW_DAYS_DEFAULT = 7
PELT_PENALTY_DEFAULT = 0.1
P_VALUE_DEFAULT = 0.05

AUTO_TRIM_FRAC_DEFAULT = 0.05


def _avg(values):
    values = [v for v in values if v is not None and not pd.isna(v)]
    return float(np.mean(values)) if values else 0.0


TOP_ACTIVITIES_DEFAULT = 15


def build_time_series(df: pd.DataFrame, numeric_attrs: list, has_start: bool,
                      window_days: int = WINDOW_DAYS_DEFAULT,
                      activity_features: list = None,
                      extend_with_log_top: bool = False):
    # Per-case aggregates (a case is one process execution in the flat setting)
    by_case = df.groupby(CASE)
    case_start = by_case[START].min()
    case_end = by_case[END].max()
    cycle_h = (case_end - case_start).dt.total_seconds() / 3600.0
    df_pairs = {}  
    for case_id, events in by_case:
        acts = events.sort_values([START, END], kind="stable")[ACTIVITY].tolist()
        df_pairs[case_id] = {(acts[i], acts[i + 1]) for i in range(len(acts) - 1)}

    service_min = (df[END] - df[START]).dt.total_seconds() / 60.0

    log_top = df[ACTIVITY].value_counts().head(TOP_ACTIVITIES_DEFAULT).index.tolist()
    if activity_features is None:
        activity_features = log_top
    elif extend_with_log_top:
        activity_features = list(activity_features) + \
            [a for a in log_top if a not in set(activity_features)]

    l_start = df[END].min()
    l_end = df[END].max()
    w = timedelta(days=window_days)
    m = int(1 + (l_end - l_start) / w)

    names_case = ["control-flow: distinct directly-follows relations",
                  "performance: avg case cycle time (h)"]
    names_event = ["resources: distinct active resources",
                   "resources: avg workload (events/resource)"]
    if has_start:
        names_event.append("performance: avg activity service time (min)")
    attr_names = {a: f"data: avg '{a}'" for a in numeric_attrs}
    if not numeric_attrs:
        names_event.append("data: number of events")
    act_share_names = {a: f"control-flow: share of activity '{a}'" for a in activity_features}
    act_service_names = {a: f"performance: avg service time (min) '{a}'"
                         for a in activity_features} if has_start else {}

    series = {n: [] for n in (names_case + names_event + list(attr_names.values())
                              + list(act_share_names.values())
                              + list(act_service_names.values()))}
    window_event_counts = []  
    time_index = []
    for i in range(m):
        ws = l_start + i * w
        we = l_start + (i + 1) * w
        time_index.append(ws)

        cases_in = case_start[(case_start >= ws) & (case_start < we)].index
        pairs = set()
        for c in cases_in:
            pairs |= df_pairs[c]
        series[names_case[0]].append(len(pairs))
        series[names_case[1]].append(_avg(cycle_h.loc[cases_in].tolist()) if len(cases_in) else 0.0)

        ev_mask = (df[END] >= ws) & (df[END] < we)
        ev = df[ev_mask]
        window_event_counts.append(len(ev))
        n_res = ev[RESOURCE].nunique()
        series["resources: distinct active resources"].append(n_res)
        series["resources: avg workload (events/resource)"].append(
            len(ev) / n_res if n_res else 0.0)
        if has_start:
            series["performance: avg activity service time (min)"].append(
                _avg(service_min[ev_mask].tolist()))
        for a, name in attr_names.items():
            series[name].append(_avg(pd.to_numeric(ev[a], errors="coerce").tolist()))
        if not numeric_attrs:
            series["data: number of events"].append(len(ev))
        act_counts = ev[ACTIVITY].value_counts()
        act_service = service_min[ev_mask].groupby(ev[ACTIVITY]).mean() \
            if has_start and len(ev) else {}
        for a in activity_features:
            series[act_share_names[a]].append(
                act_counts.get(a, 0) / len(ev) if len(ev) else 0.0)
            if has_start:
                v = act_service.get(a, np.nan) if len(ev) else 0.0
                series[act_service_names[a]].append(0.0 if pd.isna(v) else float(v))

    return ({k: np.asarray(v, dtype=float) for k, v in series.items()}, time_index,
            np.asarray(window_event_counts, dtype=float), list(activity_features))


def detect_change_points(series: dict, penalty=PELT_PENALTY_DEFAULT) -> dict:
    change_points = {}
    for name, s in series.items():
        pen = penalty.get(name, penalty.get("__default__", PELT_PENALTY_DEFAULT)) \
            if isinstance(penalty, dict) else penalty
        if len(s) < 5 or np.nanmax(np.abs(s)) == 0:
            change_points[name] = []
            continue
        normalized = s / np.nanmax(np.abs(s))
        bps = rpt.Pelt(jump=1).fit(normalized).predict(pen=pen)
        change_points[name] = [bp for bp in bps if bp < len(s)]
    return change_points


def correlate_drifts(series: dict, change_points: dict, time_index: list,
                     p_threshold: float = P_VALUE_DEFAULT) -> list:
    explainable = []
    for feat_effect in series:
        for feat_cause in series:
            if feat_effect == feat_cause:
                continue
            for d in change_points[feat_effect]:
                for d_ in change_points[feat_cause]:
                    if d_ >= d:
                        continue
                    lag = d - d_
                    try:
                        res = grangercausalitytests(
                            pd.DataFrame({feat_effect: series[feat_effect],
                                          feat_cause: series[feat_cause]}),
                            [lag], verbose=False)
                    except (ValueError, sm_exceptions.InfeasibleTestError):
                        continue  
                    p_value = res[lag][0]["ssr_ftest"][1]
                    if p_value <= p_threshold:
                        explainable.append({
                            "cause": feat_cause,
                            "effect": feat_effect,
                            "cause_window": d_,
                            "effect_window": d,
                            "cause_date": time_index[d_] if d_ < len(time_index) else None,
                            "effect_date": time_index[d] if d < len(time_index) else None,
                            "lag_windows": lag,
                            "p_value": p_value,
                        })
    explainable.sort(key=lambda e: e["p_value"])
    return explainable


def redetect(drift: dict, penalty, p_threshold: float = P_VALUE_DEFAULT) -> dict:
    change_points = detect_change_points(drift["series"], penalty)
    explainable = correlate_drifts(drift["series"], change_points,
                                   drift["time_index"], p_threshold)
    return {**drift,
            "change_points": change_points,
            "explainable_drifts": explainable,
            "params": {**drift["params"], "pelt_penalty": penalty,
                       "p_threshold": p_threshold}}


def calibrate_penalties(series: dict, pen_grid: list) -> tuple:
    penalties, residual = {}, {}
    for name, s in series.items():
        chosen = None
        for pen in sorted(pen_grid):
            if not detect_change_points({name: s}, pen)[name]:
                chosen = pen
                break
        if chosen is None:
            chosen = max(pen_grid)
            residual[name] = len(detect_change_points({name: s}, chosen)[name])
        penalties[name] = chosen
    return penalties, residual


def sparse_edge_bounds(window_event_counts, auto_trim_frac: float):
    counts = np.asarray(window_event_counts, dtype=float)
    non_empty = counts[counts > 0]
    if len(non_empty) == 0:
        return 0, len(counts)
    threshold = auto_trim_frac * np.median(non_empty)
    active = np.where(counts >= threshold)[0]
    if len(active) == 0:
        return 0, len(counts)
    return int(active[0]), int(active[-1]) + 1


def run_drift_analysis(df: pd.DataFrame, info: dict, numeric_attrs: list,
                       window_days: int = WINDOW_DAYS_DEFAULT,
                       penalty=PELT_PENALTY_DEFAULT,
                       p_threshold: float = P_VALUE_DEFAULT,
                       trim: int = 0,
                       auto_trim_frac: float = AUTO_TRIM_FRAC_DEFAULT,
                       activity_features: list = None,
                       extend_with_log_top: bool = False):
    series, time_index, counts, activities = build_time_series(
        df, numeric_attrs, info["has_start"], window_days,
        activity_features=activity_features, extend_with_log_top=extend_with_log_top)
    lo, hi = 0, len(time_index)
    if auto_trim_frac > 0:
        lo, hi = sparse_edge_bounds(counts, auto_trim_frac)
    lo, hi = lo + trim, hi - trim
    if hi - lo >= 5:  
        series = {k: s[lo:hi] for k, s in series.items()}
        time_index = time_index[lo:hi]
    change_points = detect_change_points(series, penalty)
    explainable = correlate_drifts(series, change_points, time_index, p_threshold)
    return {
        "series": series,
        "time_index": time_index,
        "change_points": change_points,
        "explainable_drifts": explainable,
        "activity_features": activities,
        "params": {"window_days": window_days, "pelt_penalty": penalty,
                   "p_threshold": p_threshold, "trim": trim,
                   "auto_trim_frac": auto_trim_frac},
    }
