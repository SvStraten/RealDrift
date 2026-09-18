from __future__ import annotations

import bisect
import warnings
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from step5_arrival_time_modeling import make_gap_source

SLOT_MINUTES_DEFAULT = 60
N_SLOTS = (7 * 24 * 60) // SLOT_MINUTES_DEFAULT


def _week_slot(ts, slot_minutes=SLOT_MINUTES_DEFAULT):
    minutes_into_week = ts.weekday() * 24 * 60 + ts.hour * 60 + ts.minute
    return int(minutes_into_week // slot_minutes) % ((7 * 24 * 60) // slot_minutes)


def discover_resource_calendars(real_log_df, case_col, act_col, time_col, res_col,
                                  complete_col=None, slot_minutes=SLOT_MINUTES_DEFAULT,
                                  min_weeks_active_frac=0.10, min_events_for_own_calendar=30,
                                  default_event_minutes=15, verbose=True):
    if res_col is None or res_col not in real_log_df.columns:
        if verbose:
            print(f"  no resource column detected -- skipping calendar discovery entirely; "
                  f"composition will proceed with NO resource-based constraint on this dataset")
        return {}, set()

    df = real_log_df.sort_values([case_col, time_col]).copy()
    if complete_col and complete_col in df.columns:
        dur_min = (pd.to_datetime(df[complete_col]) - df[time_col]).dt.total_seconds() / 60
        df["_dur_min"] = dur_min.clip(lower=1, upper=default_event_minutes * 4)
    else:
        df["_next_ts"] = df.groupby(case_col)[time_col].shift(-1)
        gap_min = (df["_next_ts"] - df[time_col]).dt.total_seconds() / 60
        df["_dur_min"] = gap_min.clip(upper=default_event_minutes).fillna(default_event_minutes).clip(lower=1)

    counts = df[res_col].value_counts()
    pooled_resources = set(counts[counts < min_events_for_own_calendar].index)
    if verbose and pooled_resources:
        print(f"  pooling {len(pooled_resources)}/{len(counts)} resources "
              f"(< {min_events_for_own_calendar} events) into per-activity shared calendars")

    span_weeks = max(1.0, (df[time_col].max() - df[time_col].min()).days / 7)
    threshold = max(1, min_weeks_active_frac * span_weeks)

    slot_hits = defaultdict(lambda: defaultdict(int))
    for row in df[[case_col, act_col, time_col, res_col, "_dur_min"]].to_dict("records"):
        res = row[res_col]
        if res is None or (isinstance(res, float) and pd.isna(res)):
            continue
        key = res if res not in pooled_resources else f"activity::{row[act_col]}"
        start_slot = _week_slot(row[time_col], slot_minutes)
        n_spanned = max(1, int(np.ceil(row["_dur_min"] / slot_minutes)))
        for k in range(n_spanned):
            slot_hits[key][(start_slot + k) % N_SLOTS] += 1

    calendars = {key: {slot for slot, cnt in hits.items() if cnt >= threshold}
                 for key, hits in slot_hits.items()}
    if verbose:
        avg_coverage = np.mean([len(c) / N_SLOTS for c in calendars.values()]) if calendars else 0
        print(f"  {len(calendars)} calendar(s) discovered, avg weekly coverage "
              f"{avg_coverage:.0%} ({slot_minutes}-min slots)")
    return calendars, pooled_resources


def _calendar_key(resource, pooled_resources):
    return f"__pooled__{resource}" if resource in pooled_resources else resource


def _in_calendar(ts, resource, activity, calendars, pooled_resources, slot_minutes=SLOT_MINUTES_DEFAULT):
    key = resource if resource not in pooled_resources else f"activity::{activity}"
    cal = calendars.get(key)
    if cal is None:
        return True 
    return _week_slot(ts, slot_minutes) in cal


def _event_specs(case, default_event_minutes=15):
    offs, acts, ress = case["offsets"], case["activities"], case["resources"]
    specs = []
    for i, (o, a, r) in enumerate(zip(offs, acts, ress)):
        if not r:
            continue
        nxt = offs[i + 1] if i + 1 < len(offs) else o + default_event_minutes * 60
        dur = max(min(nxt - o, default_event_minutes * 60), 60.0)
        specs.append((o, a, r, dur))
    return specs


def feasible(case, t0, occupancy, calendars, pooled_resources, resource_cap=1,
             default_event_minutes=15, slot_minutes=SLOT_MINUTES_DEFAULT,
             bed_occupancy=None, max_beds=None):
    specs = _event_specs(case, default_event_minutes)
    blocking = set()
    ok = True
    max_dur = pd.Timedelta(minutes=default_event_minutes)
    for o, a, r, dur in specs:
        ts = t0 + pd.to_timedelta(o, unit="s")
        if not _in_calendar(ts, r, a, calendars, pooled_resources, slot_minutes):
            ok = False
            continue  # a calendar violation cannot be fixed by eviction
        end = ts + pd.to_timedelta(dur, unit="s")
        busy = occupancy.get(r, [])
        lo = bisect.bisect_left(busy, ts - max_dur, key=lambda iv: iv[0])
        hi = bisect.bisect_right(busy, end, key=lambda iv: iv[0])
        overlapping = [(bs, be, owner) for (bs, be, owner) in busy[lo:hi] if bs < end and ts < be]
        if len(overlapping) >= resource_cap:
            ok = False
            blocking.update(owner for _, _, owner in overlapping)

    if bed_occupancy is not None and max_beds is not None:
        case_start = t0
        case_end = t0 + pd.to_timedelta(case.get("duration", 0.0), unit="s")
        lo = bisect.bisect_left(bed_occupancy, case_end, key=lambda iv: iv[0])
        bed_overlapping = [(bs, be, owner) for (bs, be, owner) in bed_occupancy[:lo]
                            if bs < case_end and case_start < be]
        if len(bed_overlapping) >= max_beds:
            ok = False
            blocking.update(owner for _, _, owner in bed_overlapping)

    return ok, blocking


def commit(case, t0, occupancy, placed, default_event_minutes=15, bed_occupancy=None):
    specs = _event_specs(case, default_event_minutes)
    case_id = case["_instance_case_key"]
    for o, a, r, dur in specs:
        ts = t0 + pd.to_timedelta(o, unit="s")
        bisect.insort(occupancy.setdefault(r, []), (ts, ts + pd.to_timedelta(dur, unit="s"), case_id),
                      key=lambda iv: iv[0])
    if bed_occupancy is not None:
        case_end = t0 + pd.to_timedelta(case.get("duration", 0.0), unit="s")
        bisect.insort(bed_occupancy, (t0, case_end, case_id), key=lambda iv: iv[0])
    placed[case_id] = (case, t0)


def uncommit(case_id, occupancy, placed, bed_occupancy=None):
    case, _ = placed.pop(case_id)
    for r in {spec[2] for spec in _event_specs(case)}:
        if r in occupancy:
            occupancy[r] = [iv for iv in occupancy[r] if iv[2] != case_id]
    if bed_occupancy is not None:
        bed_occupancy[:] = [iv for iv in bed_occupancy if iv[2] != case_id]
    return case

class TaskInstance:

    def __init__(self, label, task_id, pool, gap_source, priority, seed=0):
        rng = np.random.default_rng(seed)
        cases = list(pool)
        rng.shuffle(cases)
        self.label = label
        self.task_id = task_id
        self.queue = deque(cases)
        self.gap_source = gap_source
        self.priority = priority
        self.clock = None  # lazily anchored, see module docstring

    def __len__(self):
        return len(self.queue)

    def advance_clock(self, rng, composer_now):
        if self.clock is None:
            self.clock = composer_now
        gap = self.gap_source.sample(self.clock, rng)
        self.clock = self.clock + pd.to_timedelta(gap, unit="s")
        return self.clock


def compose_polydrift_stream(instances, drift_types, calendars, pooled_resources, t0,
                              w=30, max_preempt_evictions=5, horizon_days=21,
                              resource_cap=1, default_event_minutes=15,
                              slot_minutes=SLOT_MINUTES_DEFAULT, seed=0, verbose=True,
                              max_beds=None):
    rng = np.random.default_rng(seed)
    placed = {}
    excluded = []
    occupancy = {}
    bed_occupancy = [] if max_beds is not None else None
    transition_log = []
    raw_targets = {}

    def find_feasible_or_evict(case, target_t, active_idx):
        ok, blocking = feasible(case, target_t, occupancy, calendars, pooled_resources,
                                 resource_cap, default_event_minutes, slot_minutes,
                                 bed_occupancy, max_beds)
        if ok:
            commit(case, target_t, occupancy, placed, default_event_minutes, bed_occupancy)
            return target_t

        lower_priority_blockers = {cid for cid in blocking
                                    if cid in placed and placed[cid][0]["_instance_priority"] < instances[active_idx].priority}
        if lower_priority_blockers and len(lower_priority_blockers) <= max_preempt_evictions:
            originals = {cid: placed[cid] for cid in lower_priority_blockers}
            evicted = {cid: uncommit(cid, occupancy, placed, bed_occupancy) for cid in lower_priority_blockers}
            ok2, _ = feasible(case, target_t, occupancy, calendars, pooled_resources,
                               resource_cap, default_event_minutes, slot_minutes,
                               bed_occupancy, max_beds)
            if ok2:
                commit(case, target_t, occupancy, placed, default_event_minutes, bed_occupancy)
                for cid, ev_case in evicted.items():
                    not_before = max(target_t, raw_targets.get(cid, target_t))
                    ev_t = _reschedule_earliest_feasible(ev_case, not_before, occupancy, calendars,
                                                          pooled_resources, resource_cap,
                                                          default_event_minutes, slot_minutes,
                                                          horizon_days, bed_occupancy, max_beds)
                    if ev_t is not None:
                        commit(ev_case, ev_t, occupancy, placed, default_event_minutes, bed_occupancy)
                    else:
                        excluded.append(ev_case)
                return target_t
            for cid, ev_case in evicted.items():
                _, original_t = originals[cid]
                commit(ev_case, original_t, occupancy, placed, default_event_minutes, bed_occupancy)

        result_t = _search_forward(case, target_t, occupancy, calendars, pooled_resources,
                                    resource_cap, default_event_minutes, slot_minutes, horizon_days,
                                    bed_occupancy, max_beds)
        if result_t is not None:
            commit(case, result_t, occupancy, placed, default_event_minutes, bed_occupancy)
        return result_t

    active = 0
    first_committed = {}  # instance label -> earliest committed t (this instance's first contribution)
    last_committed = {}   # instance label -> latest committed t
    composer_clock = t0   # tracks the stream's actual progression -- see anchor comment below
    while active < len(instances):
        inst = instances[active]
        dtype = drift_types[active] if active < len(drift_types) else None

        while len(inst) > 0:
            draw_from = inst
            if dtype == "gradual" and active + 1 < len(instances):
                r_j = len(inst)
                if r_j < w:
                    p_next = 1 - r_j / w
                    if rng.random() < p_next:
                        draw_from = instances[active + 1]

            if len(draw_from) == 0:
                draw_from = inst
                if len(draw_from) == 0:
                    break

            case = draw_from.queue.popleft()
            case["_instance_case_key"] = f"{draw_from.label}_{case['case_id']}"
            case["_instance_priority"] = draw_from.priority
            case["concept"] = draw_from.label

            anchor = draw_from.clock or composer_clock
            target_t = draw_from.advance_clock(rng, anchor)
            raw_targets[case["_instance_case_key"]] = target_t

            final_t = find_feasible_or_evict(case, target_t, instances.index(draw_from))
            if final_t is not None:
                composer_clock = max(composer_clock, final_t)
                lbl = draw_from.label
                if lbl not in first_committed or final_t < first_committed[lbl]:
                    first_committed[lbl] = final_t
                if lbl not in last_committed or final_t > last_committed[lbl]:
                    last_committed[lbl] = final_t
            else:
                excluded.append(case)

        if dtype is not None and active + 1 < len(instances):
            next_label = instances[active + 1].label
            transition_log.append({
                "from": inst.label, "to": next_label, "type": dtype,
                "start_ts": first_committed.get(next_label, last_committed.get(inst.label)),
                "end_ts": last_committed.get(inst.label, first_committed.get(next_label)),
            })
        active += 1

    if verbose:
        bed_note = f", bed cap={max_beds}" if max_beds is not None else ""
        print(f"  placed {len(placed)} cases, excluded {len(excluded)} "
              f"(no feasible slot within {horizon_days}d{bed_note})")
    return placed, excluded, transition_log, occupancy, raw_targets, bed_occupancy


def _search_forward(case, start_t, occupancy, calendars, pooled_resources, resource_cap,
                     default_event_minutes, slot_minutes, horizon_days, bed_occupancy=None,
                     max_beds=None, step_minutes=15):
    """Bounded forward search for the earliest feasible t0 within horizon_days."""
    max_steps = int(horizon_days * 24 * 60 / step_minutes)
    t = start_t
    for _ in range(max_steps):
        ok, _ = feasible(case, t, occupancy, calendars, pooled_resources, resource_cap,
                          default_event_minutes, slot_minutes, bed_occupancy, max_beds)
        if ok:
            return t
        t = t + pd.Timedelta(minutes=step_minutes)
    return None


def _reschedule_earliest_feasible(case, not_before_t, occupancy, calendars, pooled_resources,
                                   resource_cap, default_event_minutes, slot_minutes, horizon_days,
                                   bed_occupancy=None, max_beds=None):
    """Used when rescheduling an evicted case: the same bounded search,
    searching only forward from not_before_t (delay, never earliness)."""
    return _search_forward(case, not_before_t, occupancy, calendars, pooled_resources,
                            resource_cap, default_event_minutes, slot_minutes, horizon_days,
                            bed_occupancy, max_beds)

def sa_refine(placed, raw_targets, occupancy, calendars, pooled_resources, resource_cap=1,              default_event_minutes=15, slot_minutes=SLOT_MINUTES_DEFAULT,
              n_iterations=2000, sample_frac=1.0, t0_temp=3600.0, cooling=0.995,
              seed=0, verbose=True, bed_occupancy=None, max_beds=None,
              horizon_days=21, shift_minutes_floor=30, shift_scale_frac=0.15):
    
    rng = np.random.default_rng(seed)
    by_instance = defaultdict(list)
    original_target = {}
    for cid, (case, t) in placed.items():
        by_instance[case["concept"]].append(cid)
        original_target[cid] = raw_targets.get(cid, t)  

    def total_delay():
        return sum((placed[cid][1] - original_target[cid]).total_seconds() for cid in placed
                   if placed[cid][1] >= original_target[cid])

    T = t0_temp
    delay0 = total_delay()
    n_sampled = max(1, int(n_iterations * sample_frac))
    all_case_ids = list(placed.keys())
    if not all_case_ids:
        return placed, 0.0

    snap_hits = 0
    for it in range(n_iterations):
        cid = all_case_ids[rng.integers(0, len(all_case_ids))]
        case, cur_t = placed[cid]
        instance_peers = by_instance[case["concept"]]
        move = rng.choice(["shift", "swap", "snap"])

        if move == "snap":
            old_case, old_t = case, cur_t
            uncommit(cid, occupancy, placed, bed_occupancy)
            new_t = _search_forward(old_case, original_target[cid], occupancy, calendars,
                                     pooled_resources, resource_cap, default_event_minutes,
                                     slot_minutes, horizon_days, bed_occupancy, max_beds)
            if new_t is not None and new_t <= old_t:
                commit(old_case, new_t, occupancy, placed, default_event_minutes, bed_occupancy)
                if new_t < old_t:
                    snap_hits += 1
            else:
                commit(old_case, old_t, occupancy, placed, default_event_minutes, bed_occupancy)
            T *= cooling
            continue

        if move == "shift" or len(instance_peers) < 2:
            cur_delay_min = max(0.0, (cur_t - original_target[cid]).total_seconds() / 60.0)
            window = max(shift_minutes_floor, int(shift_scale_frac * cur_delay_min))
            new_t = cur_t + pd.Timedelta(minutes=int(rng.integers(-window, window + 1)))
            new_t = max(new_t, original_target[cid])  # floor at own sampled target
            old_case, old_t = case, cur_t
            uncommit(cid, occupancy, placed, bed_occupancy)
            ok, _ = feasible(old_case, new_t, occupancy, calendars, pooled_resources,
                              resource_cap, default_event_minutes, slot_minutes,
                              bed_occupancy, max_beds)
            if ok:
                delay_before = max(0.0, (old_t - original_target[cid]).total_seconds())
                delay_after = max(0.0, (new_t - original_target[cid]).total_seconds())
                delta = delay_after - delay_before
                if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-6)):
                    commit(old_case, new_t, occupancy, placed, default_event_minutes, bed_occupancy)
                else:
                    commit(old_case, old_t, occupancy, placed, default_event_minutes, bed_occupancy)
            else:
                commit(old_case, old_t, occupancy, placed, default_event_minutes, bed_occupancy)
        else:
            other_cid = instance_peers[rng.integers(0, len(instance_peers))]
            if other_cid == cid:
                continue
            other_case, other_t = placed[other_cid]
            a_case, a_t = case, cur_t
            if other_t < original_target[cid] or a_t < original_target[other_cid]:
                continue

            uncommit(cid, occupancy, placed, bed_occupancy)
            uncommit(other_cid, occupancy, placed, bed_occupancy)
            ok_a, _ = feasible(a_case, other_t, occupancy, calendars, pooled_resources,
                                resource_cap, default_event_minutes, slot_minutes,
                                bed_occupancy, max_beds)
            ok_b = False
            if ok_a:
                commit(a_case, other_t, occupancy, placed, default_event_minutes, bed_occupancy)
                ok_b, _ = feasible(other_case, a_t, occupancy, calendars, pooled_resources,
                                    resource_cap, default_event_minutes, slot_minutes,
                                    bed_occupancy, max_beds)
                uncommit(cid, occupancy, placed, bed_occupancy)
            if ok_a and ok_b:
                delay_before = (max(0.0, (a_t - original_target[cid]).total_seconds()) +
                                 max(0.0, (other_t - original_target[other_cid]).total_seconds()))
                delay_after = (max(0.0, (other_t - original_target[cid]).total_seconds()) +
                                max(0.0, (a_t - original_target[other_cid]).total_seconds()))
                delta = delay_after - delay_before
                if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-6)):
                    commit(a_case, other_t, occupancy, placed, default_event_minutes, bed_occupancy)
                    commit(other_case, a_t, occupancy, placed, default_event_minutes, bed_occupancy)
                else:
                    commit(a_case, a_t, occupancy, placed, default_event_minutes, bed_occupancy)
                    commit(other_case, other_t, occupancy, placed, default_event_minutes, bed_occupancy)
            else:
                commit(a_case, a_t, occupancy, placed, default_event_minutes, bed_occupancy)
                commit(other_case, other_t, occupancy, placed, default_event_minutes, bed_occupancy)

        T *= cooling

    delay1 = total_delay()
    if verbose:
        print(f"  SA: total delay {delay0/3600:.1f}h -> {delay1/3600:.1f}h "
              f"({n_iterations} iterations, {snap_hits} snap moves improved a case, final T={T:.1f}s)")
    return placed, delay0 - delay1


def recompute_transition_log(placed, transition_log):
    first_ts_by_label = {}
    for cid, (case, t) in placed.items():
        lbl = case["concept"]
        if lbl not in first_ts_by_label or t < first_ts_by_label[lbl]:
            first_ts_by_label[lbl] = t

    fixed = []
    for tr in transition_log:
        drift_ts = first_ts_by_label.get(tr["to"])
        fixed.append({**tr, "start_ts": drift_ts, "end_ts": drift_ts})
    return fixed

def build_log_df_polydrift(placed, case_col="case:concept:name", act_col="concept:name",
                            time_col="time:timestamp", res_col="org:resource"):
    rows = []
    for case_id, (case, t0) in placed.items():
        for offset, act, res in zip(case["offsets"], case["activities"], case["resources"]):
            rows.append({
                case_col: case_id,
                act_col: act,
                time_col: t0 + pd.to_timedelta(offset, unit="s"),
                res_col: res,
                "case:concept_label": case["concept"],
            })
    return pd.DataFrame(rows).sort_values(time_col).reset_index(drop=True)
