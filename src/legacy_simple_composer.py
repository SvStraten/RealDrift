"""
legacy_simple_composer.py

Simplified drift composer (compose_simple_stream, compose_recurrent_stream)
without simulated-annealing refinement. Not used by generate_drift_log.py,
which uses step6_drift_composer.py instead. Kept for reference and quick
sanity checks on a dataset before running the full pipeline.
"""
import numpy as np
import pandas as pd
from collections import defaultdict, deque


def compose_simple_stream(segments, t0, drift_types, gradual_window=30, seed=0):
    """
    segments:     [(label, pool, gap_source), ...] in stream order
                  (gap_source = output of make_gap_source()).
    drift_types:  list of len(segments)-1, each 'sudden' or 'gradual'.
                    sudden  - concept i's queue is fully drained before
                              concept i+1 starts.
                    gradual - for `gradual_window` arrivals, each new
                              arrival is drawn from concept i with a
                              probability that decays linearly to 0 while
                              concept i+1's grows to 1, so both concepts'
                              cases interleave before the cutover.

    Returns (placed, transition_log):
      placed          - [(case_dict, anchor_timestamp), ...] in the order
                         cases were scheduled (not necessarily anchor
                         order for gradual windows).
      transition_log   - list of dicts describing each transition's
                         [start_s, end_s] window (seconds from t0) and type.
    """
    rng = np.random.default_rng(seed)
    queues, labels, sources = [], [], []
    for label, pool, gap_source in segments:
        cases = list(pool)
        rng.shuffle(cases)
        queues.append(deque(cases))
        labels.append(label)
        sources.append(gap_source)

    placed, transition_log = [], []
    t_cursor = 0.0
    active = 0
    while active < len(queues):
        dtype = drift_types[active] if active < len(drift_types) else None
        if dtype == "gradual" and active + 1 < len(queues):
            n_hold = min(gradual_window, len(queues[active]))
            while len(queues[active]) > n_hold:
                case = queues[active].popleft()
                current_ts = t0 + pd.to_timedelta(t_cursor, unit="s")
                t_cursor += sources[active].sample(current_ts, rng)
                placed.append((case, t0 + pd.to_timedelta(t_cursor, unit="s")))

            t_start = t_cursor
            n_mix = min(n_hold, len(queues[active + 1]))
            for k in range(n_mix):
                p_next = (k + 1) / (n_mix + 1)
                idx = active + 1 if rng.random() < p_next else active
                case = queues[idx].popleft()
                current_ts = t0 + pd.to_timedelta(t_cursor, unit="s")
                t_cursor += sources[idx].sample(current_ts, rng)
                placed.append((case, t0 + pd.to_timedelta(t_cursor, unit="s")))
            transition_log.append({"from": labels[active], "to": labels[active + 1],
                                    "type": "gradual", "start_s": t_start, "end_s": t_cursor})
        elif dtype == "sudden":
            transition_log.append({
                "from": labels[active],
                "to": labels[active + 1] if active + 1 < len(labels) else None,
                "type": "sudden", "start_s": t_cursor, "end_s": t_cursor,
            })
        while queues[active]:
            case = queues[active].popleft()
            current_ts = t0 + pd.to_timedelta(t_cursor, unit="s")
            t_cursor += sources[active].sample(current_ts, rng)
            placed.append((case, t0 + pd.to_timedelta(t_cursor, unit="s")))
        active += 1

    return placed, transition_log


def split_pool_recurrent(pool, seed=0):
    """Splits one concept's pool into two disjoint, shuffled halves, used
    to build a recurrent stream [C1a, C2a, ..., Cka, C1b, C2b, ..., Ckb]
    where a concept reappears without repeating the same cases."""
    rng = np.random.default_rng(seed)
    cases = list(pool)
    rng.shuffle(cases)
    half = len(cases) // 2
    return cases[:half], cases[half:]


def _event_specs(case, default_event_minutes):
    """(offset_s, resource, duration_s) for every event in `case` that has
    a resource. Duration is the gap to the next event in the same case,
    capped at `default_event_minutes`."""
    offs, ress = case["offsets"], case["resources"]
    specs = []
    for i, (o, r) in enumerate(zip(offs, ress)):
        if not r:
            continue
        nxt = offs[i + 1] if i + 1 < len(offs) else o + default_event_minutes * 60
        dur = max(min(nxt - o, default_event_minutes * 60), 60.0)
        specs.append((o, r, dur))
    return specs


def compose_recurrent_stream(segments, t0, drift_types, gradual_window=20,
                              resource_cap=1, default_event_minutes=15,
                              max_delay_steps=200, seed=0):
    """
    segments: [(label, pool, gap_source), ...], typically built by calling
      split_pool_recurrent() on each concept and chaining
      [C1a, C2a, ..., Cka, C1b, ..., Ckb].
    drift_types: len(segments)-1 list of 'sudden'/'gradual' (same semantics
      as compose_simple_stream).

    A resource can't be doing more than `resource_cap` activities at once
    (default 1). If no case in the pools has a resource value, this
    degrades to compose_simple_stream's behaviour.

    Placement: greedy delay-only -- if a case's sampled anchor would
    double-book one of its resources, the case is pushed later (in
    `default_event_minutes`-sized steps, up to `max_delay_steps`) until it
    fits, then committed. No simulated-annealing refinement.
    """
    rng = np.random.default_rng(seed)
    queues, labels, sources = [], [], []
    for label, pool, gap_source in segments:
        cases = list(pool)
        rng.shuffle(cases)
        queues.append(deque(cases))
        labels.append(label)
        sources.append(gap_source)

    has_resources = any(
        any(r for r in c["resources"]) for q in queues for c in q
    )
    if not has_resources:
        print("[legacy_simple_composer] no resource values found in any pool -- "
              "running without resource contention.")

    occupancy = defaultdict(list)  # resource -> [(start_s, end_s), ...]

    def fits(anchor_s, specs):
        for o, r, dur in specs:
            start, end = anchor_s + o, anchor_s + o + dur
            busy = occupancy[r]
            if sum(1 for (bs, be) in busy if bs < end and start < be) >= resource_cap:
                return False
        return True

    def commit(anchor_s, specs):
        for o, r, dur in specs:
            start = anchor_s + o
            occupancy[r].append((start, start + dur))

    def place(case, anchor_s):
        if not has_resources:
            return anchor_s
        specs = _event_specs(case, default_event_minutes)
        if not specs:
            return anchor_s
        t, steps = anchor_s, 0
        while not fits(t, specs) and steps < max_delay_steps:
            t += default_event_minutes * 60
            steps += 1
        commit(t, specs)
        return t

    placed, transition_log = [], []
    t_cursor = 0.0
    active = 0
    while active < len(queues):
        dtype = drift_types[active] if active < len(drift_types) else None
        if dtype == "gradual" and active + 1 < len(queues):
            n_hold = min(gradual_window, len(queues[active]))
            while len(queues[active]) > n_hold:
                case = queues[active].popleft()
                current_ts = t0 + pd.to_timedelta(t_cursor, unit="s")
                t_cursor += sources[active].sample(current_ts, rng)
                t_cursor = place(case, t_cursor)
                placed.append((case, t0 + pd.to_timedelta(t_cursor, unit="s")))

            t_start = t_cursor
            n_mix = min(n_hold, len(queues[active + 1]))
            for k in range(n_mix):
                p_next = (k + 1) / (n_mix + 1)
                idx = active + 1 if rng.random() < p_next else active
                case = queues[idx].popleft()
                current_ts = t0 + pd.to_timedelta(t_cursor, unit="s")
                t_cursor += sources[idx].sample(current_ts, rng)
                t_cursor = place(case, t_cursor)
                placed.append((case, t0 + pd.to_timedelta(t_cursor, unit="s")))
            transition_log.append({"from": labels[active], "to": labels[active + 1],
                                    "type": "gradual", "start_s": t_start, "end_s": t_cursor})
        elif dtype == "sudden":
            transition_log.append({
                "from": labels[active],
                "to": labels[active + 1] if active + 1 < len(labels) else None,
                "type": "sudden", "start_s": t_cursor, "end_s": t_cursor,
            })
        while queues[active]:
            case = queues[active].popleft()
            current_ts = t0 + pd.to_timedelta(t_cursor, unit="s")
            t_cursor += sources[active].sample(current_ts, rng)
            t_cursor = place(case, t_cursor)
            placed.append((case, t0 + pd.to_timedelta(t_cursor, unit="s")))
        active += 1

    return placed, transition_log
