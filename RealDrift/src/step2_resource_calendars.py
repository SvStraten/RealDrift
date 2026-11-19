from __future__ import annotations

from collections import Counter, deque

import pandas as pd

GRANULE_MINUTES_DEFAULT = 60
UNKNOWN_RESOURCE = "UNKNOWN"   # placeholder for events without a resource


DEFAULT_COLUMNS = {"case": "case:concept:name", "activity": "concept:name",
                   "timestamp": "time:timestamp", "resource": "org:resource",
                   "lifecycle": "lifecycle:transition"}


def load_and_prepare_log(log_path: str, columns: dict | None = None) -> pd.DataFrame:
    """Load an XES or CSV log into the columns case_id, activity, resource, timestamp and
    (when present) lifecycle. `columns` maps those roles to the file's column names, for
    example {"resource": "org:group"} for Sepsis."""
    cols = {**DEFAULT_COLUMNS, **(columns or {})}
    if log_path.lower().endswith(".xes") or log_path.lower().endswith(".xes.gz"):
        import pm4py
        df = pm4py.read_xes(log_path)
    else:
        from io_utils import _detect_separator
        df = pd.read_csv(log_path, sep=_detect_separator(log_path))

    rename = {cols["case"]: "case_id", cols["activity"]: "activity",
              cols["timestamp"]: "timestamp"}
    if cols["resource"] in df.columns:
        rename[cols["resource"]] = "resource"
    if cols["lifecycle"] in df.columns:
        rename[cols["lifecycle"]] = "lifecycle"
    df = df.rename(columns=rename)
    from io_utils import parse_timestamps
    df["timestamp"] = parse_timestamps(df["timestamp"], name="timestamp")
    if "resource" in df.columns:
        # events without a recorded resource keep the placeholder instead of being dropped
        df["resource"] = df["resource"].fillna(UNKNOWN_RESOURCE).astype(str)
    df["case_id"] = df["case_id"].astype(str)
    df["activity"] = df["activity"].astype(str)
    return df.sort_values(["case_id", "activity", "timestamp"]).reset_index(drop=True)


def scope_to_lifecycle_activities(df: pd.DataFrame) -> pd.DataFrame:
    if "lifecycle" not in df.columns:
        print("[step2] no 'lifecycle' column found -- skipping W_* scoping, "
              "every event will be treated as instantaneous")
        return df
    w_activities = df[df["lifecycle"].astype(str).str.upper() == "START"]["activity"].unique().tolist()
    if not w_activities:
        print("[step2] no activity has a START lifecycle event -- skipping W_* scoping, "
              "every event will be treated as instantaneous")
        return df
    print(f"[step2] activities with genuine start/end data: {sorted(w_activities)}")
    return df[df["activity"].isin(w_activities)].copy()


def build_activity_instances(df: pd.DataFrame) -> pd.DataFrame:
    instances = []
    has_lifecycle = "lifecycle" in df.columns
    for (case_id, activity), group in df.sort_values("timestamp").groupby(
            ["case_id", "activity"], sort=False):
        pending_starts = deque()
        for _, row in group.iterrows():
            lc = str(row["lifecycle"]).upper() if has_lifecycle else "COMPLETE"
            if lc == "START":
                pending_starts.append((row["timestamp"], row["resource"]))
            elif lc == "COMPLETE":
                if pending_starts:
                    tau_s, resource = pending_starts.popleft()
                else:
                    tau_s, resource = row["timestamp"], row["resource"]
                instances.append((case_id, activity, resource, tau_s, row["timestamp"]))
    result = pd.DataFrame(instances, columns=["case_id", "activity", "resource", "tau_s", "tau_c"])
    assert (result["tau_c"] >= result["tau_s"]).all(), "found a COMPLETE before its own START"
    return result


def gamma(ts: pd.Timestamp, n: int = GRANULE_MINUTES_DEFAULT):
    weekday = ts.day_name()
    minute_of_day = ts.hour * 60 + ts.minute
    slot_idx = minute_of_day // n
    start_minute = slot_idx * n
    tau_s_w = (start_minute // 60, start_minute % 60, 0)
    end_minute = start_minute + n
    tau_c_w = (end_minute // 60 % 24, end_minute % 60, 0)
    return (weekday, tau_s_w, tau_c_w)


def extract_calendar_entries(instances_r: pd.DataFrame, n: int = GRANULE_MINUTES_DEFAULT) -> Counter:
    omega = Counter()
    for tau_s, tau_c, activity in zip(instances_r["tau_s"], instances_r["tau_c"], instances_r["activity"]):
        es, ec = gamma(tau_s, n), gamma(tau_c, n)
        omega[(*es, activity)] += 1
        omega[(*ec, activity)] += 1
    return omega


def compute_r_participation(instances: pd.DataFrame) -> pd.Series:
    counts = instances.groupby(["resource", "activity"]).size()
    max_per_activity = counts.groupby("activity").max()
    participation = {}
    for r, r_counts in counts.groupby(level=0):
        r_counts = r_counts.droplevel(0)
        participation[r] = r_counts.sum() / max_per_activity.loc[r_counts.index].sum()
    return pd.Series(participation, name="RParticipation")


def confidence(entry_key, omega, weekday_activity_totals):
    weekday, tau_s_w, tau_c_w, activity = entry_key
    count_in_slot = omega[entry_key]
    count_on_weekday = weekday_activity_totals.get((weekday, activity), 0)
    return count_in_slot / count_on_weekday if count_on_weekday > 0 else 0.0


def support(calendar_keys: set, omega: Counter, total: int) -> float:
    if total == 0:
        return 0.0
    covered = sum(c for k, c in omega.items() if (k[0], k[1], k[2]) in calendar_keys)
    return covered / total


def discover_calendar(omega: Counter, d_supp: float, d_conf: float) -> set:
    """Algorithm 3."""
    if not omega:
        return set()
    total = sum(omega.values())
    weekday_activity_totals = Counter()
    for (w, s, e, a), c in omega.items():
        weekday_activity_totals[(w, a)] += c

    kept, discarded = set(), []
    for entry_key in omega:
        if confidence(entry_key, omega, weekday_activity_totals) >= d_conf:
            kept.add((entry_key[0], entry_key[1], entry_key[2]))
        else:
            discarded.append(entry_key)

    if support(kept, omega, total) < d_supp:
        for entry_key in sorted(discarded, key=lambda k: omega[k], reverse=True):
            slot = (entry_key[0], entry_key[1], entry_key[2])
            kept.add(slot)
            if support(kept, omega, total) >= d_supp:
                break
    return kept


def max_disjoint_intervals(events_df: pd.DataFrame) -> pd.DataFrame:
    events_sorted = events_df.sort_values("tau_s", ascending=False).reset_index(drop=True)
    kept_idx, last_start = [], None
    for idx, row in events_sorted.iterrows():
        if last_start is None or row["tau_c"] <= last_start:
            kept_idx.append(idx)
            last_start = row["tau_s"]
    return events_sorted.loc[kept_idx]


def to_composer_calendars(alloc, avail, all_resources, granule_minutes=GRANULE_MINUTES_DEFAULT):
    weekday_to_int = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                       "Friday": 4, "Saturday": 5, "Sunday": 6}
    slots_per_day = 1440 // granule_minutes

    def _to_week_slot(entry):
        weekday, (h, m, _s), _end = entry
        start_minute = h * 60 + m
        return weekday_to_int[weekday] * slots_per_day + (start_minute // granule_minutes)

    calendars = {}
    for r in alloc:
        calendars[r] = {_to_week_slot(e) for e in avail[r]}
    for key, slots in avail.items():
        if key.startswith("__joint__"):
            activity = key[len("__joint__"):]
            calendars[f"activity::{activity}"] = {_to_week_slot(e) for e in slots}

    pooled_resources = set(all_resources) - set(alloc.keys())
    return calendars, pooled_resources


def discover_resource_profiles(instances: pd.DataFrame, n: int = GRANULE_MINUTES_DEFAULT,
                                d_supp: float = 0.7, d_conf: float = 0.1, d_part: float = 0.4):
    r_participation = compute_r_participation(instances)
    avail = {}

    for r, r_instances in instances.groupby("resource"):
        if r_participation.get(r, 0.0) >= d_part:
            avail[r] = discover_calendar(extract_calendar_entries(r_instances, n), d_supp, d_conf)
        else:
            avail[r] = set()

    for activity, act_instances in instances.groupby("activity"):
        discarded = act_instances[act_instances["resource"].map(lambda r: len(avail.get(r, set())) == 0)]
        if discarded.empty:
            continue
        joint_events = max_disjoint_intervals(discarded)
        joint_name = f"__joint__{activity}"
        calendar = discover_calendar(extract_calendar_entries(joint_events, n), d_supp, d_conf)
        if calendar:
            avail[joint_name] = calendar
        else:
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            avail[joint_name] = {(d, (h, 0, 0), ((h + 1) % 24, 0, 0)) for d in days for h in range(24)}

    alloc = {r: set(instances[instances["resource"] == r]["activity"].unique())
             for r, cal in avail.items() if cal and not r.startswith("__joint__")}

    return alloc, avail, r_participation


def discover_calendars(log_path: str, columns: dict | None = None,
                       granule_minutes: int = GRANULE_MINUTES_DEFAULT,
                       d_supp: float = 0.7, d_conf: float = 0.1, d_part: float = 0.4):
    """Full calendar discovery of the paper, from the raw log to the composer format.

    Returns (calendars, pooled_resources, info). calendars maps a resource name, or
    'activity::<name>' for pooled resources, to a set of weekly slot indices.
    """
    df = load_and_prepare_log(log_path, columns)
    if "resource" not in df.columns:
        print("[step2] no resource column, composition will run without resource calendars")
        return {}, set(), {"n_calendars": 0}
    df = scope_to_lifecycle_activities(df)
    instances = build_activity_instances(df)
    instances = instances[instances["resource"] != UNKNOWN_RESOURCE]   # no calendar for a placeholder
    alloc, avail, r_participation = discover_resource_profiles(
        instances, n=granule_minutes, d_supp=d_supp, d_conf=d_conf, d_part=d_part)
    calendars, pooled = to_composer_calendars(
        alloc, avail, set(instances["resource"].unique()), granule_minutes)
    info = {"n_activity_instances": int(len(instances)), "n_resources": int(instances["resource"].nunique()),
            "n_calendars": len(calendars), "n_pooled_resources": len(pooled),
            "granule_minutes": granule_minutes, "d_supp": d_supp, "d_conf": d_conf, "d_part": d_part}
    return calendars, pooled, info


def load_or_discover_calendars(log_path: str, cache_path: str, columns: dict | None = None, **kwargs):
    """Discover once and cache as a pickle, since discovery on a full log takes a while."""
    import os
    import pickle
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        print(f"[step2] loaded cached calendars from {cache_path}")
        return payload["calendars"], payload["pooled"], payload["info"]
    calendars, pooled, info = discover_calendars(log_path, columns, **kwargs)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"calendars": calendars, "pooled": pooled, "info": info}, f)
    print(f"[step2] discovered {info.get('n_calendars', 0)} calendars, cached at {cache_path}")
    return calendars, pooled, info
