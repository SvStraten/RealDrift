"""Independent post-hoc checks of a composed log.

The composer enforces the global conditions while it builds the log. These checks recompute the
conditions from the FINAL placements (after eviction and annealing), so they also catch violations
introduced by the refinement stage. They do not call Feasible.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from step6_drift_composer import _event_specs, _in_calendar, SLOT_MINUTES_DEFAULT


def _max_concurrency(intervals):
    """Maximum number of simultaneously open intervals. An interval ending exactly when the next
    one starts does not overlap it."""
    points = []
    for s, e in intervals:
        points.append((s, 1))
        points.append((e, -1))
    points.sort(key=lambda p: (p[0], p[1]))
    cur = best = 0
    for _, d in points:
        cur += d
        best = max(best, cur)
    return best


def _merge(intervals):
    merged = []
    for s, e in sorted(intervals):
        if merged and s < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def validate_composition(placed, calendars, pooled_resources, raw_targets=None, resource_cap=1,
                         max_beds=None, default_event_minutes=15, slot_minutes=SLOT_MINUTES_DEFAULT):
    by_resource_case = defaultdict(list)   # (resource, case) -> that case's event intervals
    calendar_violations = 0
    n_events_checked = 0
    case_spans = []
    floor_violations = 0
    for cid, (case, t0) in placed.items():
        for o, a, r, dur in _event_specs(case, default_event_minutes):
            ts = t0 + pd.to_timedelta(o, unit="s")
            by_resource_case[(r, cid)].append((ts, ts + pd.to_timedelta(dur, unit="s")))
            n_events_checked += 1
            if not _in_calendar(ts, r, a, calendars, pooled_resources, slot_minutes):
                calendar_violations += 1
        case_spans.append((t0, t0 + pd.to_timedelta(case.get("duration", 0.0), unit="s")))
        if raw_targets is not None and cid in raw_targets and t0 < raw_targets[cid]:
            floor_violations += 1

    # Events of one case by the same resource can be closer together than the minimum event
    # duration. Such a self-overlap is not a conflict between cases, so intervals are merged
    # per (resource, case) before concurrency is counted across cases.
    by_resource = defaultdict(list)
    for (r, _cid), iv in by_resource_case.items():
        for s_, e_ in _merge(iv):
            by_resource[r].append((s_, e_))
    resources_over_cap = sorted(r for r, iv in by_resource.items() if _max_concurrency(iv) > resource_cap)
    result = {
        "n_cases": len(placed),
        "n_events_checked": n_events_checked,
        "resource_cap": resource_cap,
        "resources_over_cap": resources_over_cap,
        "calendar_violations": calendar_violations,
        "floor_violations": floor_violations,
        "max_concurrent_cases": _max_concurrency(case_spans) if case_spans else 0,
        "max_beds": max_beds,
        "bed_cap_violated": bool(max_beds is not None and case_spans and _max_concurrency(case_spans) > max_beds),
    }
    result["ok"] = (not resources_over_cap and calendar_violations == 0 and floor_violations == 0
                    and not result["bed_cap_violated"])
    return result


def validate_ground_truth(transition_log, placed):
    """The recorded transitions must agree with the final placements."""
    first = {}
    for cid, (case, t) in placed.items():
        lbl = case["concept"]
        if lbl not in first or t < first[lbl]:
            first[lbl] = t
    problems = []
    for tr in transition_log:
        if tr["start_ts"] is None or tr["end_ts"] is None:
            problems.append(f"{tr['from']}->{tr['to']} has no timestamp")
            continue
        if tr["start_ts"] > tr["end_ts"]:
            problems.append(f"{tr['from']}->{tr['to']} start after end")
        if tr["to"] in first and tr["type"] == "sudden" and tr["start_ts"] != first[tr["to"]]:
            problems.append(f"{tr['from']}->{tr['to']} sudden drift not at first start of incoming instance")
        if tr["to"] in first and not (tr["start_ts"] <= first[tr["to"]] <= tr["end_ts"]):
            problems.append(f"{tr['from']}->{tr['to']} interval does not contain first start of incoming instance")
    return {"n_transitions": len(transition_log), "problems": problems, "ok": not problems}
