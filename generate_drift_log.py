
from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from io_utils import load_pool
from step5_arrival_time_modeling import make_gap_source
from step6_drift_composer import (
    TaskInstance,
    compose_polydrift_stream,
    sa_refine,
    recompute_transition_log,
    build_log_df_polydrift,
)


def discover_ranks(concept_pools_dir, dataset):
    """Finds generated_{dataset}_rank{N}{A,B}.csv files and returns
    {rank: {'A': path, 'B': path or None}}, sorted by rank."""
    pattern = os.path.join(concept_pools_dir, f"generated_{dataset}_rank*.csv")
    found = {}
    for p in glob.glob(pattern):
        m = re.search(rf"generated_{dataset}_rank(\d+)([AB])\.csv$", os.path.basename(p))
        if not m:
            continue
        rank, instance = int(m.group(1)), m.group(2)
        found.setdefault(rank, {})[instance] = p
    return dict(sorted(found.items()))


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tier", choices=["sudden", "gradual", "recurrent"], default="recurrent",
                         help="sudden/gradual: one instance per concept, chained with that "
                              "transition type. recurrent: two instances (A/B) per concept, "
                              "all transitions sudden. See module docstring.")
    parser.add_argument("--calendars", default=None,
                         help="Optional pickle of (calendars, pooled_resources) from "
                              "step2_resource_calendars.py. If omitted, composition runs "
                              "with no resource-based feasibility constraint.")
    parser.add_argument("--max-beds", type=int, default=None,
                         help="Optional global bed-capacity condition (subsection 4.4).")
    parser.add_argument("--skip-sa", action="store_true",
                         help="Skip the simulated annealing refinement stage (subsection 4.5) "
                              "for a faster, greedy-only run.")
    parser.add_argument("--sa-iterations", type=int, default=2000)
    parser.add_argument("--gradual-window", type=int, default=30,
                         help="blend window w (cases) for 'gradual' transitions -- larger w means "
                              "blending starts earlier and the transition is more spread out. "
                              "Has no effect on 'sudden' transitions.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--max-cases-per-instance", type=int, default=None,
                         help="Truncate each instance's pool to this many cases -- useful "
                              "for a quick smoke test before running the full dataset. "
                              "Without a resource calendar, full-scale BPIC12 composition "
                              "can take a long time; see README.md.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["dataset"]
    sublogs_dir = cfg["sublogs_dir"]
    concept_pools_dir = cfg["concept_pools_dir"]
    t0 = pd.Timestamp(cfg["t0"])
    use_atkde = cfg.get("use_atkde", False)

    os.makedirs(args.out_dir, exist_ok=True)

    ranks = discover_ranks(concept_pools_dir, dataset)
    if not ranks:
        raise SystemExit(f"No generated_{dataset}_rank*.csv files found under {concept_pools_dir}")
    print(f"Found {len(ranks)} concept(s) for {dataset}: ranks {list(ranks.keys())}")

    calendars, pooled_resources = {}, set()
    if args.calendars:
        with open(args.calendars, "rb") as f:
            calendars, pooled_resources = pickle.load(f)
        print(f"Loaded {len(calendars)} resource calendar(s) from {args.calendars}")
    else:
        print("No --calendars given -- composing WITHOUT a resource-based feasibility "
              "constraint. See README.md for how to add one.")

    # --- fit one arrival model (gap source) per task, from the REAL sublog ---
    gap_sources = {}
    for rank in ranks:
        sublog_path = os.path.join(sublogs_dir, f"{dataset}_sublog_C{rank}.csv")
        if not os.path.exists(sublog_path):
            raise SystemExit(f"Missing real sublog for rank {rank}: {sublog_path}")
        _, arrivals = load_pool(sublog_path, concept_label=f"C{rank}", verbose=False)
        gap_sources[rank] = make_gap_source(arrivals, use_atkde=use_atkde)
        print(f"  rank {rank}: fit arrival model on {len(arrivals)} real arrivals "
              f"({arrivals.min()} .. {arrivals.max()})")

    # --- build the TaskInstance sequence, per --tier ---
    instances = []
    priority = 0
    instance_letters = ("A",) if args.tier in ("sudden", "gradual") else ("A", "B")
    for instance_letter in instance_letters:
        for rank in ranks:
            path = ranks[rank].get(instance_letter)
            if path is None:
                print(f"  WARNING: rank {rank} has no {instance_letter} instance, skipping")
                continue
            pool, _ = load_pool(path, concept_label=f"C{rank}{instance_letter}", verbose=False)
            if args.max_cases_per_instance is not None:
                pool = pool[: args.max_cases_per_instance]
            for case in pool:
                case["_instance_case_key"] = f"{case['concept']}_{case['case_id']}"
                case["_instance_priority"] = priority
            instances.append(TaskInstance(
                label=f"C{rank}{instance_letter}", task_id=rank, pool=pool,
                gap_source=gap_sources[rank], priority=priority, seed=args.seed + priority,
            ))
            priority += 1
    print(f"[{args.tier}] Composed instance sequence: {[inst.label for inst in instances]}")

    if args.tier == "recurrent":
        drift_types = ["sudden"] * (len(instances) - 1)
    else:
        drift_types = [args.tier] * (len(instances) - 1)  # 'sudden' or 'gradual', every transition

    # --- stage 1: greedy composition ---
    placed, excluded, transition_log, occupancy, raw_targets, bed_occupancy = compose_polydrift_stream(
        instances, drift_types, calendars, pooled_resources, t0,
        w=args.gradual_window, max_beds=args.max_beds, seed=args.seed,
    )
    print(f"Greedy pass: {len(placed)} placed, {len(excluded)} excluded")

    # --- stage 2: simulated annealing refinement ---
    if not args.skip_sa:
        placed, delay_improvement = sa_refine(
            placed, raw_targets, occupancy, calendars, pooled_resources,
            n_iterations=args.sa_iterations, bed_occupancy=bed_occupancy, max_beds=args.max_beds,
            seed=args.seed,
        )
        print(f"SA refinement improved total delay by {delay_improvement/3600:.1f}h")

    transition_log = recompute_transition_log(placed, transition_log)

    # --- export ---
    log_df = build_log_df_polydrift(placed)
    out_log_path = os.path.join(args.out_dir, f"{dataset}_{args.tier}_drift_log.csv")
    log_df.to_csv(out_log_path, index=False)
    print(f"Wrote {out_log_path}  ({log_df['case:concept:name'].nunique()} cases, {len(log_df)} events)")


    out_transitions_path = os.path.join(args.out_dir, f"{dataset}_{args.tier}_transitions.csv")
    pd.DataFrame([
        {"from": tr["from"], "to": tr["to"], "type": tr["type"],
         "start_ts": tr["start_ts"], "end_ts": tr["end_ts"]}
        for tr in transition_log
    ]).to_csv(out_transitions_path, index=False)
    print(f"Wrote {out_transitions_path}")

    if excluded:
        excluded_path = os.path.join(args.out_dir, f"{dataset}_{args.tier}_excluded.csv")
        pd.DataFrame([{"case_id": c["case_id"], "concept": c["concept"]} for c in excluded]).to_csv(
            excluded_path, index=False)
        print(f"Wrote {excluded_path}  ({len(excluded)} cases excluded, could not be scheduled feasibly)")


if __name__ == "__main__":
    main()
