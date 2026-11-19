import xml.etree.ElementTree as ET

import pandas as pd
import pm4py

CASE = "case"
ACTIVITY = "activity"
START = "start_time"
END = "end_time"
RESOURCE = "resource"

STANDARD_EVENT_KEYS = {
    "concept:name", "time:timestamp", "lifecycle:transition", "org:resource",
    "start_timestamp", "time:start", "start:timestamp",
}
START_TIME_KEYS = ("start_timestamp", "start:timestamp", "time:start")
STANDARD_TRACE_KEYS = {"concept:name", "variant", "variant-index", "creator"}


def read_global_attribute_keys(xes_path: str) -> dict:
    """Parse the XES header and return {'event': [keys], 'trace': [keys]} of <global> declarations."""
    found = {"event": [], "trace": []}
    for ev, elem in ET.iterparse(xes_path, events=("start", "end")):
        tag = elem.tag.split("}")[-1]
        if ev == "end" and tag == "global":
            scope = elem.attrib.get("scope", "event")
            keys = [child.attrib.get("key") for child in elem if child.attrib.get("key")]
            found.setdefault(scope, []).extend(keys)
            elem.clear()
        elif ev == "start" and tag == "trace":
            break  # header is over, stop parsing
    return found


def _pair_lifecycle(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([CASE, END], kind="stable")
    lc = df["lifecycle:transition"].astype(str).str.lower()
    starts = {}  # (case, activity) -> list of pending start timestamps
    start_times = []
    keep_mask = []
    for case_id, activity, end_ts, transition in zip(df[CASE], df[ACTIVITY], df[END], lc):
        key = (case_id, activity)
        if transition == "start":
            starts.setdefault(key, []).append(end_ts)
            keep_mask.append(False)
        else:
            pending = starts.get(key)
            start_times.append(pending.pop(0) if pending else end_ts)
            keep_mask.append(True)
    out = df[pd.Series(keep_mask, index=df.index)].copy()
    out[START] = start_times
    return out


def load_xes(xes_path: str) -> tuple:
    df = pm4py.read_xes(xes_path)
    df = df.rename(columns={
        "case:concept:name": CASE,
        "concept:name": ACTIVITY,
        "time:timestamp": END,
        "org:resource": RESOURCE,
    })
    df[END] = pd.to_datetime(df[END], utc=True)

    has_start = True
    start_col = next((c for c in START_TIME_KEYS if c in df.columns), None)
    if start_col is not None:
        df[START] = pd.to_datetime(df[start_col], utc=True)
        df = df.drop(columns=[start_col])
    elif "lifecycle:transition" in df.columns and \
            (df["lifecycle:transition"].astype(str).str.lower() == "start").any():
        df = _pair_lifecycle(df)
    else:
        df[START] = df[END]
        has_start = False
    if "lifecycle:transition" in df.columns:
        df = df.drop(columns=["lifecycle:transition"])

    has_resource = RESOURCE in df.columns and df[RESOURCE].notna().any()
    if not has_resource:
        df[RESOURCE] = "UNKNOWN"

    df = df.sort_values([CASE, START, END], kind="stable").reset_index(drop=True)

    globals_ = read_global_attribute_keys(xes_path)
    info = {
        "global_event_attrs": [k for k in globals_["event"] if k not in STANDARD_EVENT_KEYS],
        "global_trace_attrs": [k for k in globals_["trace"] if k not in STANDARD_TRACE_KEYS],
        "has_start": has_start,
        "has_resource": has_resource,
    }
    return df, info


def data_attribute_columns(df: pd.DataFrame, info: dict) -> list:
    declared = list(info["global_event_attrs"]) + \
        ["case:" + k for k in info["global_trace_attrs"]]
    cols = [c for c in declared if c in df.columns]
    if not cols:  # no <global> declarations: fall back to all attribute columns
        reserved = {CASE, ACTIVITY, START, END, RESOURCE} | STANDARD_EVENT_KEYS \
            | {"case:" + k for k in STANDARD_TRACE_KEYS}
        cols = [c for c in df.columns if c not in reserved]
    return cols
