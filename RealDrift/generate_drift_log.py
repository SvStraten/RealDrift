"""Compose one synthetic event log with controlled drift. See README.md."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from pipeline import TIERS, compose_run, load_config, write_run  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--tier", choices=TIERS, default="recurrent",
                   help="sudden or gradual, one instance per task. hard, the instances A and B of each "
                        "task with sudden transitions between them and gradual ones between tasks. "
                        "recurrent, A of all tasks then B of all tasks with sudden transitions.")
    p.add_argument("--calendars", default="auto",
                   help="auto (discover from the raw log in the config and cache), none, or a pickle path")
    p.add_argument("--max-beds", type=int, default=None, help="optional global bed-capacity condition")
    p.add_argument("--skip-sa", action="store_true", help="greedy pass only")
    p.add_argument("--sa-iterations", type=int, default=None, help="overrides the config (default 3000)")
    p.add_argument("--gradual-window", type=int, default=None, help="blend window w (default 30)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--export-xes", action="store_true", help="also write an XES file for the detectors")
    p.add_argument("--max-cases-per-instance", type=int, default=None, help="truncate pools for a smoke test")
    a = p.parse_args()

    cfg = load_config(a.config)
    if a.max_beds is not None:
        cfg["composer"]["max_beds"] = a.max_beds
    if a.sa_iterations is not None:
        cfg["composer"]["sa_iterations"] = a.sa_iterations
    if a.gradual_window is not None:
        cfg["composer"]["gradual_window"] = a.gradual_window

    result = compose_run(cfg, a.tier, a.seed, calendars_mode=a.calendars, skip_sa=a.skip_sa,
                         max_cases_per_instance=a.max_cases_per_instance)
    paths = write_run(result, a.out_dir, export_xes_flag=a.export_xes)
    m = result["manifest"]
    print(f"placed {m['n_placed']}, excluded {m['n_excluded']}, mean delay {m['mean_delay_per_case_h']} h")
    print(f"validation ok={m['validation']['ok']}, ground truth ok={m['ground_truth_check']['ok']}")
    for k, v in paths.items():
        print(f"wrote {k}: {v}")
    if not (m["validation"]["ok"] and m["ground_truth_check"]["ok"]):
        sys.exit("validation failed, see manifest")


if __name__ == "__main__":
    main()
