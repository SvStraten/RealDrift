"""Compose every dataset x tier x seed of configs/experiment.yaml and summarize the manifests.

    python scripts/run_composition.py                      # evaluation seeds
    python scripts/run_composition.py --phase tune         # tuning seeds for the detectors
    python scripts/run_composition.py --datasets bpic12 --tiers sudden --seeds 0 1

Existing streams are reused. Pass --force to recompose them (needed after any code or parameter change).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pipeline import compose_run, load_config, write_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="configs/experiment.yaml")
    ap.add_argument("--phase", choices=["eval", "tune"], default="eval")
    ap.add_argument("--datasets", nargs="*", default=None, help="dataset names to run, default all")
    ap.add_argument("--tiers", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-xes", action="store_true")
    ap.add_argument("--max-cases-per-instance", type=int, default=None)
    args = ap.parse_args()

    os.chdir(ROOT)
    with open(args.experiment) as f:
        exp = yaml.safe_load(f)
    tiers = args.tiers or exp["tiers"]
    seeds = args.seeds if args.seeds is not None else exp["seeds" if args.phase == "eval" else "tuning_seeds"]
    out_dir = os.path.join(exp["out_dir"], args.phase)

    rows, failures = [], []
    for cfg_path in exp["configs"]:
        cfg = load_config(cfg_path)
        if args.datasets and cfg["dataset"] not in args.datasets:
            continue
        pools = os.path.join(cfg["concept_pools_dir"], f"generated_{cfg['dataset']}_rank1A.csv")
        if not os.path.exists(pools):
            print(f"skip {cfg['dataset']}, no generated pools under {cfg['concept_pools_dir']}")
            continue
        for tier in tiers:
            for seed in seeds:
                prefix = f"{cfg['dataset']}_{tier}_seed{seed}"
                mpath = os.path.join(out_dir, f"{prefix}_manifest.json")
                if os.path.exists(mpath) and not args.force:
                    manifest = json.load(open(mpath))
                else:
                    res = compose_run(cfg, tier, seed, calendars_mode=exp.get("calendars", "auto"),
                                      max_cases_per_instance=args.max_cases_per_instance, verbose=False)
                    write_run(res, out_dir, export_xes_flag=not args.no_xes)
                    manifest = res["manifest"]
                m = manifest
                ok = m["validation"]["ok"] and m["ground_truth_check"]["ok"]
                if not ok:
                    failures.append(prefix)
                rows.append({"dataset": m["dataset"], "tier": m["tier"], "seed": m["seed"],
                             "n_placed": m["n_placed"], "n_excluded": m["n_excluded"],
                             "excluded_share": m["n_excluded"] / max(1, m["n_placed"] + m["n_excluded"]),
                             "mean_delay_per_case_h": m["mean_delay_per_case_h"],
                             "delay_greedy_total_h": m["delay_greedy_total_h"],
                             "delay_final_total_h": m["delay_final_total_h"],
                             "n_drifts": len(m["drift_types"]),
                             "atkde_fallback_instances": len(m["arrival_fallback_instances"]),
                             "valid": ok, "runtime_s": m["runtime_s"]})
                print(f"{prefix}: placed {m['n_placed']} excluded {m['n_excluded']} valid {ok}")
    if rows:
        os.makedirs(out_dir, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, "composition_summary.csv"), index=False)
        print(df.groupby(["dataset", "tier"])[["n_excluded", "excluded_share", "mean_delay_per_case_h"]]
              .mean().round(3).to_string())
    if failures:
        sys.exit(f"validation failed for {failures}")


if __name__ == "__main__":
    main()
