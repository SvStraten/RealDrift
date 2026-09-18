"""
step6_drift_composer.py

Resource-calendar discovery format conversion, the feasibility predicate
Feasible() (subsection 4.4), and drift-stream composition (subsection
4.5): greedy placement with preemption, followed by simulated-annealing
refinement.

Composition follows one procedure; what varies between runs is which
task instances are chained in what order, and each transition's type
(sudden or gradual). "Recurrent" is not a third drift type, it is a
structural property: the same task's pool used more than once as an
independently-seeded instance, matching the `rank{i}{A,B}.csv` naming
(rank{i} = task T_i's pool, A/B = two independently-seeded instances of
that task).

Notes:

  1. Per-instance arrival clocks are lazily anchored: an instance's own
     arrival model only starts ticking once that instance is first drawn
     from, anchored at the composer's current position at that moment,
     not from the global t0. This mirrors atkde_adapter.py's own
     lazy-anchoring design.
  2. The per-resource occupancy structure is kept sorted by start time,
     and feasibility checks use bisect to find the small neighborhood of
     entries that could overlap a candidate event, rather than scanning
     the full history for that resource (see feasible()'s own docstring).
"""
from __future__ import annotations

import bisect
import warnings
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from step5_arrival_time_modeling import make_gap_source

# --------------------------------------------------------------------------
# Section 4.1 -- resource calendar discovery
# --------------------------------------------------------------------------

SLOT_MINUTES_DEFAULT = 60
N_SLOTS = (7 * 24 * 60) // SLOT_MINUTES_DEFAULT


def _week_slot(ts, slot_minutes=SLOT_MINUTES_DEFAULT):
    """Weekly slot index (Monday 00:00 = slot 0), independent of calendar date."""
    minutes_into_week = ts.weekday() * 24 * 60 + ts.hour * 60 + ts.minute
    return int(minutes_into_week // slot_minutes) % ((7 * 24 * 60) // slot_minutes)


def discover_resource_calendars(real_log_df, case_col, act_col, time_col, res_col,
                                  complete_col=None, slot_minutes=SLOT_MINUTES_DEFAULT,
                                  min_weeks_active_frac=0.10, min_events_for_own_calendar=30,
                                  default_event_minutes=15, verbose=True):
    """Weekly availability calendar per resource (subsection 4.1),
    a simpler slot-hit-count alternative to step2_resource_calendars.py's
    confidence/support method. Assumes paired (case, activity, resource,
    start, complete) tuples when `complete_col` is given; otherwise falls
    back to a heuristic duration (gap to the next event in the same case,
    capped at `default_event_minutes`).

    Resources with fewer than `min_events_for_own_calendar` total events
    are pooled into a shared calendar per activity.

    Returns (calendars, pooled_resources):
      calendars        - {key: set(slot_indices)}, key = resource name, or
                          "activity::<name>" for pooled resources
      pooled_resources  - set of resource names that got pooled

    If `res_col` is None or not present in the log, calendar discovery
    has nothing to discover from and returns ({}, set()) with a printed
    warning. Downstream, `feasible()`/`_in_calendar()` treat an
    unknown/missing calendar key as "no calendar info, do not block",
    and `_event_specs()` skips events with no resource, so composition
    degrades gracefully to running without any resource-based
    constraint.
    """
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
    # itertuples() mangles column names with ':' in them into invalid attribute
    # names -- use .to_dict('records') instead so arbitrary XES-style column
    # names (case:concept:name, org:resource, ...) work unmodified
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
        return True  # no calendar info for this resource -- documented: don't block on unknowns
    return _week_slot(ts, slot_minutes) in cal


# --------------------------------------------------------------------------
# Section 4.4 -- feasibility (global condition G)
# --------------------------------------------------------------------------

def _event_specs(case, default_event_minutes=15):
    """(offset_s, activity, resource, duration_s) for every resourced event in a case."""
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
    """Feasible(c, t0 | G, S), subsection 4.4 -- checked against every
    event of `case`, not only its first, since a resource conflict
    several events into a case is as much a violation of G as one at
    case start.

    t0: pd.Timestamp candidate start time.
    occupancy: {resource: [(start_ts, end_ts, owner_case_id), ...]} of
      already-committed events, the per-resource condition. Must be kept
      sorted by start_ts (commit()/uncommit() below maintain this
      invariant); this function bisects to the small neighborhood of
      entries that could overlap a candidate event, rather than scanning
      every event ever committed to that resource. Since _event_specs()
      bounds every event's duration to at most default_event_minutes,
      only committed events whose own start falls within one
      default_event_minutes window of the candidate's [ts, end) can
      possibly overlap it, so bisect finds that window in O(log n)
      instead of an O(n) scan.
    bed_occupancy / max_beds: optional log-wide capacity condition (e.g.
      the number of beds in a hospital, imposing a capacity limit on how
      many cases may draw on it at once). Unlike the resource conditions,
      this is case-level, not event-level: a case occupies "a bed" for
      its whole span (first event to last event), and adding it must not
      push the number of concurrently active cases, log-wide across
      every task/instance, above max_beds at any point during that span.
      bed_occupancy: list of (start_ts, end_ts, owner_case_id) for every
      already-committed case's span, also kept sorted by start_ts. A
      case's span is not bounded the way a single event's duration is,
      so this bisects only the lower edge and scans forward from there.
      Both must be given together; if either is None, the bed condition
      is skipped.

    Returns (is_feasible: bool, blocking: set of owner_case_ids, either
    resource- or bed-blocking, used by the caller to find preemption
    candidates when infeasible).
    """
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
        # entries at index >= lo start after case_end and cannot overlap it, so this
        # trims the upper end of the scan; cases can be arbitrarily long, so the lower
        # portion is still scanned in full
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
    """Removes case_id from occupancy/bed_occupancy/placed. Only touches
    the resources this case actually used (via its own event specs)
    rather than every resource in occupancy. Each touched resource's list
    stays sorted by start_ts (a
    filter preserves order), maintaining feasible()'s bisect invariant."""
    case, _ = placed.pop(case_id)
    for r in {spec[2] for spec in _event_specs(case)}:
        if r in occupancy:
            occupancy[r] = [iv for iv in occupancy[r] if iv[2] != case_id]
    if bed_occupancy is not None:
        bed_occupancy[:] = [iv for iv in bed_occupancy if iv[2] != case_id]
    return case


# --------------------------------------------------------------------------
# Section 4.5 -- drift stream composition
# --------------------------------------------------------------------------

class TaskInstance:
    """One occurrence of a task's pool in the composed sequence (Sec 4.5).
    Carries the task's arrival model f_i and a priority equal to its
    position in the sequence (not its task identity) -- so a later instance
    of the SAME task still outranks an earlier instance of a DIFFERENT task."""

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
    """Subsection 4.5: draws one candidate at a time, cycling through
    instances as their pools empty; the transition type is realized via
    the selection rule; feasibility is checked with preemption (evict a
    small set of blocking lower-priority committed cases and reschedule
    them) or a bounded forward search; a case is excluded if neither
    succeeds.

    instances: list of TaskInstance, in sequence order.
    drift_types: len(instances)-1 list of 'sudden'/'gradual'.
    t0: pd.Timestamp, overall stream start, used to anchor the first
        instance's arrival clock (later instances anchor lazily to
        wherever the composer's wall-clock is when first drawn from, see
        TaskInstance).
    max_beds: optional int, a global bed-capacity condition. If given, a
        case is only feasible if committing it would not push the number
        of concurrently active cases (log-wide, across every instance,
        not per-resource) above max_beds at any point during the case's
        own span. Preemption/eviction and the bounded forward search
        both respect this the same way they respect resource conditions.
        None (default) means no bed constraint.

    Returns (placed: {case_id: (case, t)}, excluded: [case], transition_log,
             occupancy, raw_targets: {case_id: pd.Timestamp}, bed_occupancy).
             raw_targets is each case's arrival-model-sampled time before
             any feasibility delay, eviction, or search, the target
             sa_refine floors against, not the stage-1 committed time.
             bed_occupancy is returned so sa_refine
             can respect the same bed constraint during refinement.
    """
    rng = np.random.default_rng(seed)
    placed = {}
    excluded = []
    occupancy = {}
    bed_occupancy = [] if max_beds is not None else None
    transition_log = []
    raw_targets = {}

    def find_feasible_or_evict(case, target_t, active_idx):
        """Try target_t; if infeasible, try evicting a small set of blocking
        cases from a lower-priority instance; else search forward within
        horizon_days; else return None (case excluded).

        Commits `case` itself in every success path rather than leaving
        that to the caller. This matters for the eviction path
        specifically: evicted cases are rescheduled via a search that
        needs to see the evictor's own span already reserved in
        occupancy/bed_occupancy, or they can be placed right back into
        overlap with it, which would let final concurrent-case counts
        exceed a configured bed cap even though every individual
        feasibility check passed.
        """
        ok, blocking = feasible(case, target_t, occupancy, calendars, pooled_resources,
                                 resource_cap, default_event_minutes, slot_minutes,
                                 bed_occupancy, max_beds)
        if ok:
            commit(case, target_t, occupancy, placed, default_event_minutes, bed_occupancy)
            return target_t

        lower_priority_blockers = {cid for cid in blocking
                                    if cid in placed and placed[cid][0]["_instance_priority"] < instances[active_idx].priority}
        if lower_priority_blockers and len(lower_priority_blockers) <= max_preempt_evictions:
            # save (case, original_t) BEFORE uncommitting -- placed[cid] is gone after
            # uncommit(), so this is the only chance to remember where to restore to
            # if the eviction attempt turns out not to help
            originals = {cid: placed[cid] for cid in lower_priority_blockers}
            evicted = {cid: uncommit(cid, occupancy, placed, bed_occupancy) for cid in lower_priority_blockers}
            ok2, _ = feasible(case, target_t, occupancy, calendars, pooled_resources,
                               resource_cap, default_event_minutes, slot_minutes,
                               bed_occupancy, max_beds)
            if ok2:
                # commit the EVICTOR first, before rescheduling the evicted
                # cases -- see docstring above for why this ordering matters
                commit(case, target_t, occupancy, placed, default_event_minutes, bed_occupancy)
                for cid, ev_case in evicted.items():
                    # search forward from whichever is LATER: the evictor's
                    # target, or the evicted case's own raw sampled target --
                    # searching from just target_t alone can reschedule the
                    # evicted case to a time before its own floor if the
                    # evictor's target happens to be earlier, since a later
                    # event in the evicted case (not its anchor) can be what
                    # actually conflicted.
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
            # eviction didn't actually free things up (e.g. blocked by calendar too) --
            # restore evicted cases to their ORIGINAL committed time, not target_t
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
            # --- selection rule realizing drift_j ---
            draw_from = inst
            if dtype == "gradual" and active + 1 < len(instances):
                r_j = len(inst)
                if r_j < w:
                    p_next = 1 - r_j / w
                    if rng.random() < p_next:
                        draw_from = instances[active + 1]
            # sudden: instance j+1 never drawn from until instance j is fully empty (handled by outer loop)

            if len(draw_from) == 0:
                draw_from = inst
                if len(draw_from) == 0:
                    break

            case = draw_from.queue.popleft()
            case["_instance_case_key"] = f"{draw_from.label}_{case['case_id']}"
            case["_instance_priority"] = draw_from.priority
            case["concept"] = draw_from.label

            # Anchor this instance's clock: its own last tick if it has
            # ticked before; otherwise the composer's current progression
            # (composer_clock, tracked from the latest committed time
            # across all instances so far), not t0. A brand-new instance
            # starting to contribute (e.g. C2 right after C1's queue
            # drains under a sudden transition) has never ticked, so its
            # clock is None, and must anchor to composer_clock rather
            # than restarting at the beginning of the stream.
            anchor = draw_from.clock or composer_clock
            target_t = draw_from.advance_clock(rng, anchor)
            raw_targets[case["_instance_case_key"]] = target_t

            # find_feasible_or_evict already commits `case` on success (see its
            # docstring) -- do NOT commit again here, that would double-insert
            # into occupancy/bed_occupancy and corrupt the capacity accounting.
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
            # ground-truth transition instant: [last time the draining instance
            # contributed a case, first time the next instance did] -- degenerates
            # to a single instant for 'sudden' (next instance's first commit only
            # happens once the draining one is fully empty, so these coincide or
            # sit right next to each other); for 'gradual' this is a genuine window.
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


# --------------------------------------------------------------------------
# Section 4.5 -- simulated annealing refinement (second stage)
# --------------------------------------------------------------------------

def sa_refine(placed, raw_targets, occupancy, calendars, pooled_resources, resource_cap=1,              default_event_minutes=15, slot_minutes=SLOT_MINUTES_DEFAULT,
              n_iterations=2000, sample_frac=1.0, t0_temp=3600.0, cooling=0.995,
              seed=0, verbose=True, bed_occupancy=None, max_beds=None,
              horizon_days=21, shift_minutes_floor=30, shift_scale_frac=0.15):
    """Subsection 4.5, second stage: simulated annealing (Bertsimas &
    Tsitsiklis 1993), restricted to moves within a single task instance.
    Shift/swap/snap moves on sampled cases, each floored at its own
    originally-sampled target (delay only, never earliness). A move is
    accepted if it reduces total delay, else accepted with probability
    exp(-delta/T); T decays geometrically.

    Move types:

      1. 'shift' scales its candidate window with the case's own current
         delay (shift_scale_frac of it, floored at shift_minutes_floor
         minutes), so a case delayed by 10 days gets a much wider
         candidate window than one delayed by 20 minutes.
      2. 'snap': uncommit the case and run the same bounded forward
         search the greedy stage uses (_search_forward), starting from
         the case's own raw target instead of its current position.
         Since other cases may have moved since stage 1 committed this
         one, the earliest feasible slot from the case's own target can
         be earlier than where stage 1 first found room for it. This is
         accepted greedily (if feasible and no worse) rather than via
         the Metropolis criterion, since it is already an expensive
         deterministic search rather than a random perturbation.

    raw_targets: {case_id: pd.Timestamp} from compose_polydrift_stream,
    each case's arrival-model-sampled time before any stage-1 delay.
    This, not the stage-1 committed time, is the target every move must
    floor against; using the stage-1 output would re-baseline delay
    against an already-delayed time and understate it.

    bed_occupancy / max_beds: same bed-capacity condition as
    compose_polydrift_stream. Pass the bed_occupancy list returned by
    that call (not a fresh one) so SA's moves are checked against the
    same log-wide state stage 1 already built, and max_beds must match
    what was used there. A move that would violate the bed cap is
    rejected the same way a resource-infeasible move already is.

    horizon_days: bound for the 'snap' move's forward search, same
    parameter compose_polydrift_stream's stage 1 uses.

    Returns updated `placed` dict (in place) and total delay reduction (s).
    """
    rng = np.random.default_rng(seed)
    by_instance = defaultdict(list)
    original_target = {}
    for cid, (case, t) in placed.items():
        by_instance[case["concept"]].append(cid)
        original_target[cid] = raw_targets.get(cid, t)  # fall back to t only if truly untracked

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

            # HARD floor check before attempting the swap -- a swap that
            # would place either case earlier than its own raw sampled
            # target is not a candidate move at all (delay only, never
            # earliness). The delay_before/delay_after computation below
            # clamps the delay metric to zero, which is not the same as
            # forbidding the move; this floor check is what actually
            # prevents it.
            if other_t < original_target[cid] or a_t < original_target[other_cid]:
                continue

            uncommit(cid, occupancy, placed, bed_occupancy)
            uncommit(other_cid, occupancy, placed, bed_occupancy)
            ok_a, _ = feasible(a_case, other_t, occupancy, calendars, pooled_resources,
                                resource_cap, default_event_minutes, slot_minutes,
                                bed_occupancy, max_beds)
            # a_case is tentatively committed at other_t before checking ok_b, so
            # B's feasibility check sees A's new position too -- checking both
            # cases against occupancy while both are still uncommitted would let
            # each look individually feasible against "everyone else" while
            # conflicting with each other post-swap. The tentative commit is
            # undone either way; the actual commit (of both, or neither) happens
            # below once the accept/reject decision is made.
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
    """Recomputes each transition's timestamp from the actual final case
    placements in `placed`, after sa_refine has run.

    The drift point is when the new ('to') concept starts flowing in,
    the first arrival of any case belonging to that concept, read from
    the actual post-refinement placements (sa_refine can shift which
    case ends up first for a concept, so a pre-refinement snapshot can
    go stale). start_ts and end_ts are both set to this single point
    (kept as two fields for compatibility with downstream code that
    expects a window, but they are always equal). A transition is the
    moment the incoming concept begins, not tied to when the outgoing
    concept's lingering long-running cases happen to finish generating
    events, which is a separate question.
    """
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


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

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
