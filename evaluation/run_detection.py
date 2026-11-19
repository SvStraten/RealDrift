"""Run the configured detectors on the composed streams and score them.

    python -m evaluation.run_detection --phase tune     # grid search on the tuning seeds
    python -m evaluation.run_detection --phase eval     # report on the evaluation seeds

Tuning chooses, per detector and dataset, the grid point with the best mean F1 over all tiers of
the tuning streams and writes outputs/tuned_params.yaml. The evaluation phase uses these values.
Outputs of the evaluation phase are outputs/detection_raw.csv (one row per stream, detector and
tolerance) and outputs/detection_summary.csv (mean and sample standard deviation over seeds).
"""
from __future__ import annotations

import argparse
import glob
import importlib
import itertools
import json
import os
import sys

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation import detectors as D  # noqa: E402
from evaluation.scoring import load_ground_truth, resolve_tolerance, score_detections  # noqa: E402


def stream_span_seconds(log_csv):
    """Time from the first to the last event of a stream, read from its drift log."""
    head = pd.read_csv(log_csv, nrows=1)
    col = "time:timestamp" if "time:timestamp" in head.columns else next(c for c in head.columns if "time" in c.lower())
    t = pd.to_datetime(pd.read_csv(log_csv, usecols=[col])[col], format="ISO8601", utc=True)
    return float((t.max() - t.min()).total_seconds())


def discover_streams(streams_dir):
    streams = []
    for m in sorted(glob.glob(os.path.join(streams_dir, "*_manifest.json"))):
        prefix = m[:-len("_manifest.json")]
        man = json.load(open(m))
        xes, gt = prefix + ".xes", prefix + "_transitions.csv"
        if os.path.exists(xes) and os.path.exists(gt):
            streams.append({"dataset": man["dataset"], "tier": man["tier"], "seed": man["seed"],
                            "xes": xes, "gt": gt, "span_seconds": stream_span_seconds(prefix + "_drift_log.csv")})
    return streams


def merged_params(det_cfg, dataset, override=None):
    p = dict(det_cfg.get("params") or {})
    p.update((det_cfg.get("per_dataset") or {}).get(dataset) or {})
    p.update(override or {})
    return p


def run_one(name, stream, params, tolerances, gt_cache):
    gt = gt_cache.setdefault(stream["gt"], load_ground_truth(stream["gt"]))
    rows = []
    try:
        res = D.get(name)(stream["xes"], params)
        error = None
    except Exception as e:  # a failing detector run is recorded, not hidden
        res, error = None, f"{type(e).__name__}: {e}"
    for tol in tolerances:
        base = {"dataset": stream["dataset"], "tier": stream["tier"], "seed": stream["seed"],
                "detector": name, "tolerance": tol["name"], "n_gt": len(gt)}
        if res is None:
            rows.append({**base, "status": "error", "error": error, "n_det": 0, "tp": 0, "fp": 0,
                         "fn": len(gt), "precision": 0.0, "recall": 0.0, "f1": 0.0})
            continue
        s = score_detections(res.timestamps, gt, resolve_tolerance(tol, res.window_seconds, stream.get("span_seconds")))
        rows.append({**base, "status": "ok", "error": None, "n_det": s.n_det, "tp": s.tp, "fp": s.fp,
                     "fn": s.fn, "precision": s.precision, "recall": s.recall, "f1": s.f1,
                     "mean_distance_days": sum(s.distances_days) / len(s.distances_days) if s.distances_days else None})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["eval", "tune"], default="eval")
    ap.add_argument("--detectors", default="configs/detectors.yaml")
    ap.add_argument("--experiment", default="configs/experiment.yaml")
    ap.add_argument("--tuned", default="outputs/tuned_params.yaml")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    os.chdir(ROOT)
    cfg = yaml.safe_load(open(args.detectors))
    exp = yaml.safe_load(open(args.experiment))
    detectors = cfg.get("detectors") or {}
    if not detectors:
        raise SystemExit("no detectors configured in " + args.detectors)
    for name, dc in detectors.items():
        importlib.import_module(dc["module"])
    tolerances = cfg["tolerances"]
    streams = discover_streams(os.path.join(exp["out_dir"], args.phase))
    if not streams:
        raise SystemExit(f"no streams with xes and ground truth under {exp['out_dir']}/{args.phase}, "
                         f"run scripts/run_composition.py --phase {args.phase} first")
    gt_cache = {}

    if args.phase == "tune":
        tuned = {}
        for name, dc in detectors.items():
            for dataset in sorted({s["dataset"] for s in streams}):
                # a dataset can replace entries of the grid (per_dataset_grid), for example the
                # number of windows of a small log
                grid = {**(dc.get("grid") or {}), **((dc.get("per_dataset_grid") or {}).get(dataset) or {})}
                keys = list(grid)
                combos = [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])] or [{}]
                best, best_f1 = None, -1.0
                for combo in combos:
                    rows = []
                    for s in [s for s in streams if s["dataset"] == dataset]:
                        rows += run_one(name, s, merged_params(dc, dataset, combo), tolerances[:1], gt_cache)
                    f1 = sum(r["f1"] for r in rows) / max(1, len(rows))
                    print(f"[tune] {name}/{dataset} {combo} mean F1 {f1:.3f}")
                    if f1 > best_f1:
                        best, best_f1 = combo, f1
                tuned.setdefault(name, {})[dataset] = {**best, "tuning_mean_f1": round(best_f1, 4)}
        os.makedirs(os.path.dirname(args.tuned), exist_ok=True)
        yaml.safe_dump(tuned, open(args.tuned, "w"))
        print(f"wrote {args.tuned}")
        return

    tuned = yaml.safe_load(open(args.tuned)) if os.path.exists(args.tuned) else {}
    rows = []
    for name, dc in detectors.items():
        for s in streams:
            override = {k: v for k, v in (tuned.get(name, {}).get(s["dataset"]) or {}).items()
                        if k != "tuning_mean_f1"}
            rows += run_one(name, s, merged_params(dc, s["dataset"], override), tolerances, gt_cache)
    raw = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    raw.to_csv(os.path.join(args.out_dir, "detection_raw.csv"), index=False)

    use = raw if cfg.get("errors_count_as_failure", True) else raw[raw["status"] == "ok"]
    g = use.groupby(["dataset", "tier", "detector", "tolerance"])
    summary = g.agg(n_runs=("seed", "count"), n_errors=("status", lambda s: int((s == "error").sum())),
                    n_gt=("n_gt", "first"),
                    precision_mean=("precision", "mean"), precision_std=("precision", "std"),
                    recall_mean=("recall", "mean"), recall_std=("recall", "std"),
                    f1_mean=("f1", "mean"), f1_std=("f1", "std")).reset_index()
    summary.to_csv(os.path.join(args.out_dir, "detection_summary.csv"), index=False)
    print(summary[summary["tolerance"] == tolerances[0]["name"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
