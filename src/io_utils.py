"""Shared CSV/XES loading and export utilities."""
from __future__ import annotations

import re
import warnings

import pandas as pd

CASE_COL_CANDIDATES = ["case:concept:name", "case_id", "Case ID", "CaseID", "case"]
ACT_COL_CANDIDATES = ["concept:name", "activity", "Activity", "concept_name"]
TIME_COL_CANDIDATES = ["time:timestamp", "timestamp", "Timestamp", "Complete Timestamp",
                        "time_timestamp"]
RES_COL_CANDIDATES = ["org:resource", "resource", "Resource", "org_resource", "org:group"]
LABEL_COL_CANDIDATES = ["case:concept_label", "concept_label", "case:label", "label"]

_LIFECYCLE_SUFFIX_RE = re.compile(r"-(?:SCHEDULE|START|COMPLETE)$")


def parse_timestamps(values, name="timestamp") -> pd.Series:
    """Parse a column of timestamp strings to UTC datetimes without silently losing rows.

    pandas infers the parse format from the first non-null value. A column that mixes
    values with and without fractional seconds (as the BPIC2012 sublogs do) then turns
    every non-matching value into NaT. We therefore parse as ISO8601 first and fall back
    to the slower format="mixed", and raise if anything stays unparsable.
    """
    series = pd.Series(values)
    for fmt in ("ISO8601", "mixed"):
        try:
            parsed = pd.to_datetime(series, utc=True, format=fmt, errors="coerce")
        except (ValueError, TypeError):
            continue
        if parsed.isna().sum() == series.isna().sum():
            return parsed
    n_bad = int(parsed.isna().sum() - series.isna().sum())
    raise ValueError(f"{n_bad} {name} value(s) could not be parsed, for example "
                     f"{series[parsed.isna() & series.notna()].head(3).tolist()}")


def _keep_complete_lifecycle_only(df: pd.DataFrame, act_col: str, csv_path: str, verbose: bool) -> pd.DataFrame:
    """Generated pools and some exports encode the lifecycle in the activity name
    ('A_SUBMITTED-COMPLETE'). Keep the COMPLETE rows and strip the suffix."""
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
    """Load one sublog or generated pool.

    Returns (pool, arrivals). pool is a list of case dicts with the keys case_id, offsets
    (seconds from the first event), duration, activities, resources and concept.
    arrivals is the sorted Series of the first-event timestamps of all cases.
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
    df[time_col] = parse_timestamps(df[time_col], name=f"{csv_path} timestamp")
    df = df.sort_values([case_col, time_col])

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


def export_xes(log_df, path, case_col="case:concept:name"):
    import pm4py
    pm4py.write_xes(log_df, path, case_id_key=case_col)
    print(f"wrote {path}  ({log_df[case_col].nunique()} cases, {len(log_df)} events)")
