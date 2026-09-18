"""
io_utils.py

Shared CSV loading and export utilities used by trace pool generation
output, arrival fitting, drift composition, and final log export.

Column names are auto-detected among common XES-export variants (see
CASE_COL_CANDIDATES etc. below). The loader raises a clear KeyError
naming the columns it actually found if none of the candidates match.
"""
import re
import warnings

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

CASE_COL_CANDIDATES = ["case:concept:name", "case_id", "Case ID", "CaseID", "case"]
ACT_COL_CANDIDATES = ["concept:name", "activity", "Activity", "concept_name"]
TIME_COL_CANDIDATES = ["time:timestamp", "timestamp", "Timestamp", "Complete Timestamp",
                        "time_timestamp"]
RES_COL_CANDIDATES = ["org:resource", "resource", "Resource", "org_resource", "org:group"]
LABEL_COL_CANDIDATES = ["case:concept_label", "concept_label", "case:label", "label"]

_LIFECYCLE_SUFFIX_RE = re.compile(r"-(?:SCHEDULE|START|COMPLETE)$")


def _keep_complete_lifecycle_only(df: pd.DataFrame, act_col: str, csv_path: str, verbose: bool) -> pd.DataFrame:
    """Generated concept pools carry SCHEDULE/START/COMPLETE lifecycle rows
    per activity (e.g. 'A_SUBMITTED-COMPLETE'), with SCHEDULE/START rows
    sometimes carrying a different or 'UNKNOWN' resource than the COMPLETE
    row for the same activity execution. Detects the suffix pattern and,
    only if present, filters to COMPLETE rows and strips the suffix. A log
    with no suffixed activity names (e.g. the real sublogs) passes through
    unchanged."""
    activities = df[act_col].astype(str)
    if not activities.str.contains(_LIFECYCLE_SUFFIX_RE, regex=True).any():
        return df
    before = len(df)
    df = df[activities.str.endswith("-COMPLETE")].copy()
    df[act_col] = df[act_col].astype(str).str.replace(_LIFECYCLE_SUFFIX_RE, "", regex=True)
    if verbose:
        print(f"[{csv_path}] detected SCHEDULE/START/COMPLETE lifecycle suffixes -- "
              f"kept {len(df)}/{before} COMPLETE rows, dropped the rest, stripped the suffix")
    return df


def _pick(df, candidates, required=True, label=""):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"couldn't find a '{label}' column among {candidates} -- "
            f"the file has columns {list(df.columns)}. Pass an explicit "
            f"rename dict to load_pool(), e.g. rename={{'case_col': '...'}}."
        )
    return None


def _detect_separator(csv_path):
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    return ";" if first_line.count(";") > first_line.count(",") else ","


def load_pool(csv_path, concept_label=None, rename=None, verbose=True):
    """Load one per-concept sublog CSV.

    Returns (pool, arrival_times):
      pool           - list of case dicts: {case_id, offsets (s, from case
                       start), duration, activities, resources, concept}
      arrival_times  - sorted pandas Series of each case's REAL first-event
                       timestamp, straight from the file -- this is what
                       fit_flat_kde/ATKDESampler should be fit on, since it
                       reflects this concept's true arrival rate.
    """
    rename = rename or {}
    sep = _detect_separator(csv_path)
    df = pd.read_csv(csv_path, sep=sep)

    case_col = rename.get("case_col") or _pick(df, CASE_COL_CANDIDATES, label="case id")
    act_col = rename.get("activity_col") or _pick(df, ACT_COL_CANDIDATES, label="activity")
    time_col = rename.get("time_col") or _pick(df, TIME_COL_CANDIDATES, label="timestamp")
    res_col = rename.get("resource_col") or _pick(df, RES_COL_CANDIDATES, required=False,
                                                   label="resource")

    if verbose:
        print(f"[{csv_path}] sep='{sep}' -> case='{case_col}' activity='{act_col}' "
              f"time='{time_col}' resource='{res_col}'  ({len(df)} rows)")

    df = _keep_complete_lifecycle_only(df, act_col, csv_path, verbose)

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    n_bad = df[time_col].isna().sum()
    if n_bad:
        warnings.warn(f"{csv_path}: dropping {n_bad} rows with unparsable timestamps")
    df = df.dropna(subset=[time_col]).sort_values([case_col, time_col])

    pool, arrivals = [], []
    for case_id, grp in df.groupby(case_col, sort=False):
        grp = grp.sort_values(time_col).reset_index(drop=True)
        t0 = grp[time_col].iloc[0]
        offsets = (grp[time_col] - t0).dt.total_seconds().to_numpy()
        resources = grp[res_col].tolist() if res_col else [None] * len(grp)
        pool.append({
            "case_id": case_id,
            "offsets": offsets,
            "duration": float(offsets[-1] - offsets[0]),
            "activities": grp[act_col].astype(str).tolist(),
            "resources": [None if pd.isna(r) else str(r) for r in resources],
            "concept": concept_label,
        })
        arrivals.append(t0)

    arrivals = pd.Series(sorted(arrivals))
    if verbose:
        print(f"  -> {len(pool)} cases, real arrival span "
              f"{arrivals.min()} .. {arrivals.max()}")
    return pool, arrivals


def load_all_concepts(paths_by_label, rename=None, verbose=True):
    """paths_by_label: {'C1': 'emergency_sublog_C1.csv', ...} (ordered dict
    recommended -- concept order matters for composition). Returns
    {'C1': (pool, arrivals), ...}."""
    out = {}
    for label, path in paths_by_label.items():
        out[label] = load_pool(path, concept_label=label, rename=rename, verbose=verbose)
    return out


def build_log_df(placed, case_col="case:concept:name", act_col="concept:name",
                  time_col="time:timestamp", res_col="org:resource"):
    """placed: [(case_dict, anchor_timestamp), ...] -> flat event-log
    DataFrame, sorted by timestamp, with a unique case id per placement
    (concept label + original case id, since the same real case id can
    recur across the recurrent tier's two halves)."""
    rows = []
    for case, anchor in placed:
        new_case_id = f"{case['concept']}_{case['case_id']}"
        for offset, act, res in zip(case["offsets"], case["activities"], case["resources"]):
            rows.append({
                case_col: new_case_id,
                act_col: act,
                time_col: anchor + pd.to_timedelta(offset, unit="s"),
                res_col: res,
                "case:concept_label": case["concept"],
            })
    df = pd.DataFrame(rows).sort_values(time_col).reset_index(drop=True)
    return df


def export_xes(log_df, path, case_col="case:concept:name"):
    import pm4py
    pm4py.write_xes(log_df, path, case_id_key=case_col)
    print(f"wrote {path}  ({log_df[case_col].nunique()} cases, {len(log_df)} events)")


def export_transitions(transition_log, t0, path):
    rows = [{
        "from": tr["from"], "to": tr["to"], "type": tr["type"],
        "start_ts": (t0 + pd.to_timedelta(tr["start_s"], unit="s")).isoformat(),
        "end_ts": (t0 + pd.to_timedelta(tr["end_s"], unit="s")).isoformat(),
    } for tr in transition_log]
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    print(f"wrote {path}")
    return out
