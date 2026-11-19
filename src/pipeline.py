"""Composition pipeline shared by generate_drift_log.py, the experiment runner and the tests."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import yaml

from io_utils import load_pool, export_xes
from step5_arrival_time_modeling import make_gap_source
from step6_drift_composer import (TaskInstance, compose_polydrift_stream, sa_refine,
                                  recompute_transition_log, build_log_df_polydrift)
from validation import validate_composition, validate_ground_truth

TIERS = ("sudden", "gradual", "hard", "recurrent")

COMPOSER_DEFAULTS = {"gradual_window": 30, "horizon_days": 21, "max_preempt_evictions": 5,
                     "sa_iterations": 3000, "resource_cap": 1, "max_beds": None,
                     "default_event_minutes": 15}


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["composer"] = {**COMPOSER_DEFAULTS, **(cfg.get("composer") or {})}
    return cfg


def discover_ranks(concept_pools_dir, dataset):
    """{rank: {'A': path, 'B': path}} from generated_{dataset}_rank{N}{A,B}.csv."""
    import glob
    import re
    found = {}
    for p in glob.glob(os.path.join(concept_pools_dir, f"generated_{dataset}_rank*.csv")):
        m = re.search(rf"generated_{re.escape(dataset)}_rank(\d+)([AB])\.csv$", os.path.basename(p))
        if m:
            found.setdefault(int(m.group(1)), {})[m.group(2)] = p
    return dict(sorted(found.items()))


def instance_plan(tier, ranks):
    """Ordered list of (rank, letter) and the drift type of every transition.

    sudden, gradual   one instance (A) per task, every transition of that type
    recurrent         A of every task, then B of every task, all transitions sudden
    hard              A and B of each task in turn (R1A, R1B, R2A, ...), sudden between the two
                      instances of one task and gradual at every task boundary
    """
    if tier in ("sudden", "gradual"):
        plan = [(r, "A") for r in ranks]
        types = [tier] * (len(plan) - 1)
    elif tier == "recurrent":
        plan = [(r, l) for l in ("A", "B") for r in ranks]
        types = ["sudden"] * (len(plan) - 1)
    elif tier == "hard":
        plan = [(r, l) for r in ranks for l in ("A", "B")]
        types = ["sudden" if plan[i][0] == plan[i + 1][0] else "gradual" for i in range(len(plan) - 1)]
    else:
        raise ValueError(f"unknown tier {tier!r}, expected one of {TIERS}")
    return plan, types


def resolve_calendars(cfg, mode, verbose=True):
    """mode is 'auto' (discover from cfg['raw_log'] and cache), 'none', or a pickle path."""
    if mode == "none":
        return {}, set(), {"mode": "none"}
    if mode == "auto":
        from step2_resource_calendars import load_or_discover_calendars
        if not cfg.get("raw_log") or not os.path.exists(cfg["raw_log"]):
            raise SystemExit(
                f"calendars=auto needs the raw log ({cfg.get('raw_log')!r} not found). Provide it, pass a "
                f"calendar pickle with --calendars PATH, or pass --calendars none to compose without "
                f"resource calendars. Without calendars the global conditions are not enforced.")
        cp = cfg.get("calendar_params") or {}
        cache = cfg.get("calendars_cache") or f"data/calendars/{cfg['dataset']}_calendars.pkl"
        calendars, pooled, info = load_or_discover_calendars(cfg["raw_log"], cache, cfg.get("columns"), **cp)
        return calendars, pooled, {"mode": "auto", **info}
    import pickle
    with open(mode, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict):
        calendars, pooled = payload["calendars"], payload["pooled"]
    else:
        calendars, pooled = payload
    return calendars, pooled, {"mode": "pickle", "path": mode}


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    except Exception:
        return None


def compose_run(cfg, tier, seed, calendars_mode="auto", skip_sa=False, max_cases_per_instance=None,
                verbose=True, calendars=None):
    """Compose one drift log. Returns a dict with the log, transitions, excluded cases and manifest.

    Seeds. The composition uses 2*seed and the annealing 2*seed+1, so the two stages draw from
    independent streams. Instance k (priority k) shuffles its pool with 1000*seed+k and fits its
    own AT-KDE sampler seeded with 1000*seed+k. The trained decoders and generated pools are
    fixed across seeds, only the composition varies.
    """
    t_start = time.time()
    comp = cfg["composer"]
    dataset = cfg["dataset"]
    use_atkde = cfg.get("use_atkde", True)
    strict = cfg.get("atkde_strict", False)

    ranks = discover_ranks(cfg["concept_pools_dir"], dataset)
    if not ranks:
        raise SystemExit(f"No generated_{dataset}_rank*.csv files under {cfg['concept_pools_dir']}")
    if cfg.get("n_clusters") is not None and len(ranks) != cfg["n_clusters"]:
        raise SystemExit(f"config n_clusters={cfg['n_clusters']} but found {len(ranks)} rank(s) {list(ranks)}")

    if calendars is None:
        calendars, pooled_resources, cal_info = resolve_calendars(cfg, calendars_mode, verbose)
    else:
        calendars, pooled_resources = calendars
        cal_info = {"mode": "given"}

    plan, drift_types = instance_plan(tier, list(ranks))
    arrivals_by_rank = {}
    for rank in ranks:
        sub = os.path.join(cfg["sublogs_dir"], f"{dataset}_sublog_C{rank}.csv")
        if not os.path.exists(sub):
            raise SystemExit(f"Missing real sublog for rank {rank}: {sub}")
        _, arrivals_by_rank[rank] = load_pool(sub, concept_label=f"C{rank}", verbose=False)

    # t0 anchors the first arrival. "auto" (the default) is the earliest real case start, which
    # lies inside the time range AT-KDE was fitted on.
    t0_cfg = cfg.get("t0", "auto")
    if t0_cfg in (None, "auto"):
        t0 = min(a.min() for a in arrivals_by_rank.values())
    else:
        t0 = pd.Timestamp(t0_cfg)
        t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0.tz_convert("UTC")

    instances, gap_sources = [], []
    for priority, (rank, letter) in enumerate(plan):
        path = ranks[rank].get(letter)
        if path is None:
            raise SystemExit(f"rank {rank} has no {letter} pool but tier {tier!r} needs it")
        pool, _ = load_pool(path, concept_label=f"C{rank}{letter}", verbose=False)
        if max_cases_per_instance is not None:
            pool = pool[:max_cases_per_instance]
        unconstrained = set(cfg.get("unconstrained_resources") or [])
        if unconstrained:
            for case in pool:
                case["constraint_resources"] = [None if r in unconstrained else r for r in case["resources"]]
        gs = make_gap_source(arrivals_by_rank[rank], use_atkde=use_atkde,
                             seed=1000 * seed + priority, strict=strict)
        gap_sources.append(gs)
        instances.append(TaskInstance(label=f"C{rank}{letter}", task_id=rank, pool=pool, gap_source=gs,
                                      priority=priority, seed=1000 * seed + priority))
    if verbose:
        print(f"[{dataset}/{tier}/seed{seed}] sequence {[i.label for i in instances]} "
              f"drifts {drift_types}")

    placed, excluded, transition_log, occupancy, raw_targets, bed_occupancy = compose_polydrift_stream(
        instances, drift_types, calendars, pooled_resources, t0, w=comp["gradual_window"],
        max_preempt_evictions=comp["max_preempt_evictions"], horizon_days=comp["horizon_days"],
        resource_cap=comp["resource_cap"], default_event_minutes=comp["default_event_minutes"],
        seed=2 * seed, verbose=verbose, max_beds=comp["max_beds"])

    def total_delay_h():
        return sum(max(0.0, (t - raw_targets[c]).total_seconds()) for c, (_, t) in placed.items()
                   if c in raw_targets) / 3600

    delay_greedy_h = total_delay_h()
    sa_iterations = 0 if skip_sa else comp["sa_iterations"]
    if not skip_sa:
        placed, _ = sa_refine(placed, raw_targets, occupancy, calendars, pooled_resources,
                              resource_cap=comp["resource_cap"],
                              default_event_minutes=comp["default_event_minutes"],
                              n_iterations=sa_iterations, bed_occupancy=bed_occupancy,
                              max_beds=comp["max_beds"], horizon_days=comp["horizon_days"],
                              seed=2 * seed + 1, verbose=verbose)
    delay_final_h = total_delay_h()

    transition_log = recompute_transition_log(placed, transition_log)
    log_df = build_log_df_polydrift(placed)

    validation = validate_composition(placed, calendars, pooled_resources, raw_targets,
                                      resource_cap=comp["resource_cap"], max_beds=comp["max_beds"],
                                      default_event_minutes=comp["default_event_minutes"])
    gt_check = validate_ground_truth(transition_log, placed)

    gap_info = {inst.label: gs.describe() for inst, gs in zip(instances, gap_sources)}
    fallbacks = [l for l, d in gap_info.items() if d["requested_atkde"] and d["n_flat_samples"] > 0]
    if fallbacks and verbose:
        print(f"  WARNING: flat-KDE fallback used for {fallbacks}, see manifest for reasons")

    n_cases = max(1, len(placed))
    manifest = {
        "dataset": dataset, "tier": tier, "seed": seed,
        "sequence": [i.label for i in instances], "drift_types": drift_types,
        "parameters": {**comp, "sa_iterations": sa_iterations,
                       "unconstrained_resources": sorted(cfg.get("unconstrained_resources") or []), "t0": t0.isoformat(),
                       "use_atkde": use_atkde, "atkde_strict": strict, "n_clusters": len(ranks)},
        "calendars": {**cal_info, "n_calendars": len(calendars), "n_pooled_resources": len(pooled_resources)},
        "arrival_models": gap_info,
        "arrival_fallback_instances": fallbacks,
        "n_placed": len(placed), "n_excluded": len(excluded),
        "cases_per_instance": {i.label: sum(1 for c, (cs, _) in placed.items() if cs["concept"] == i.label)
                               for i in instances},
        "delay_greedy_total_h": round(delay_greedy_h, 3),
        "delay_final_total_h": round(delay_final_h, 3),
        "mean_delay_per_case_h": round(delay_final_h / n_cases, 4),
        "validation": validation, "ground_truth_check": gt_check,
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": np.__version__, "git_commit": _git_commit()},
        "runtime_s": round(time.time() - t_start, 1),
    }
    return {"log": log_df, "transitions": transition_log, "excluded": excluded,
            "placed": placed, "manifest": manifest}


def transitions_frame(transition_log):
    return pd.DataFrame([{
        "drift_id": i, "from": tr["from"], "to": tr["to"], "type": tr["type"],
        "start_ts": tr["start_ts"].isoformat() if tr["start_ts"] is not None else None,
        "end_ts": tr["end_ts"].isoformat() if tr["end_ts"] is not None else None,
        "n_overlap": tr.get("n_overlap", 0)} for i, tr in enumerate(transition_log)])


def write_run(result, out_dir, export_xes_flag=False, prefix=None):
    m = result["manifest"]
    prefix = prefix or f"{m['dataset']}_{m['tier']}_seed{m['seed']}"
    os.makedirs(out_dir, exist_ok=True)
    paths = {"log": os.path.join(out_dir, f"{prefix}_drift_log.csv"),
             "transitions": os.path.join(out_dir, f"{prefix}_transitions.csv"),
             "manifest": os.path.join(out_dir, f"{prefix}_manifest.json")}
    result["log"].to_csv(paths["log"], index=False)
    transitions_frame(result["transitions"]).to_csv(paths["transitions"], index=False)
    if export_xes_flag:
        paths["xes"] = os.path.join(out_dir, f"{prefix}.xes")
        export_xes(result["log"], paths["xes"])
    if result["excluded"]:
        paths["excluded"] = os.path.join(out_dir, f"{prefix}_excluded.csv")
        pd.DataFrame([{"case_id": c["case_id"], "concept": c["concept"]} for c in result["excluded"]]
                     ).to_csv(paths["excluded"], index=False)
    with open(paths["manifest"], "w") as f:
        json.dump(m, f, indent=2, default=str)
    return paths
