
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import step2_resource_calendars as s2  
from io_utils import load_pool, parse_timestamps  
from pipeline import load_config, discover_ranks  
from step6_drift_composer import _event_specs, _in_calendar  


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--d-supp", type=float, nargs="+", default=[0.7, 0.9, 0.95, 0.99, 1.0])
    ap.add_argument("--sample", type=int, default=2000, help="number of real cases to test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.chdir(ROOT)
    cfg = load_config(args.config)
    dataset = cfg["dataset"]
    cp = dict(cfg.get("calendar_params") or {})
    unconstrained = set(cfg.get("unconstrained_resources") or [])

    df = s2.load_and_prepare_log(cfg["raw_log"], cfg.get("columns"))
    df = s2.scope_to_lifecycle_activities(df)
    instances = s2.build_activity_instances(df)
    instances = instances[instances["resource"] != s2.UNKNOWN_RESOURCE]

    cases, starts = [], {}
    for rank in discover_ranks(cfg["concept_pools_dir"], dataset):
        path = os.path.join(cfg["sublogs_dir"], f"{dataset}_sublog_C{rank}.csv")
        pool, _ = load_pool(path, concept_label=f"C{rank}", verbose=False)
        raw = pd.read_csv(path)
        raw["ts"] = parse_timestamps(raw[[c for c in raw.columns if c in ("timestamp", "time:timestamp")][0]])
        id_col = "case_id" if "case_id" in raw.columns else "case:concept:name"
        starts.update(raw.groupby(id_col)["ts"].min().rename(lambda x: str(x)).to_dict())
        for c in pool:
            c["constraint_resources"] = [None if r in unconstrained else r for r in c["resources"]]
        cases += pool
    rng = np.random.default_rng(args.seed)
    rng.shuffle(cases)
    cases = cases[:args.sample]

    rows = []
    for d_supp in args.d_supp:
        alloc, avail, _ = s2.discover_resource_profiles(
            instances, n=cp.get("granule_minutes", 60), d_supp=d_supp,
            d_conf=cp.get("d_conf", 0.1), d_part=cp.get("d_part", 0.4))
        cal, pooled = s2.to_composer_calendars(alloc, avail, set(instances["resource"].unique()),
                                               cp.get("granule_minutes", 60))
        n_ev = n_in = n_ok = 0
        for c in cases:
            t0 = starts[str(c["case_id"])]
            ok = True
            for o, a, r, _d in _event_specs(c):
                n_ev += 1
                if _in_calendar(t0 + pd.to_timedelta(o, unit="s"), r, a, cal, pooled):
                    n_in += 1
                else:
                    ok = False
            n_ok += ok
        rows.append({"d_supp": d_supp, "events_inside": n_in / max(n_ev, 1),
                     "cases_fully_inside": n_ok / len(cases),
                     "avg_weekly_coverage": float(np.mean([len(v) / 168 for v in cal.values()])) if cal else 0.0})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    os.makedirs("outputs", exist_ok=True)
    out.to_csv(f"outputs/{dataset}_calendar_diagnostics.csv", index=False)


if __name__ == "__main__":
    main()
