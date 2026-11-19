"""Raw event log -> features -> clusters -> sublogs -> resource calendars.

    python scripts/prepare_dataset.py --config configs/bpic2012_config.yaml

Steps
  1. load the raw log (XES or CSV) using the column roles of the config
  2. keep the events used for clustering (config clustering.keep_lifecycle, "complete" or "all")
  3. build the case attributes of config clustering.case_attributes (log1p or category codes)
  4. trace features and K-means with the FIXED number of clusters n_clusters (step3.cluster_fixed_k)
  5. write one sublog per cluster, C1 being the largest, plus the case to cluster table
  6. discover and cache the resource calendars (step2), used by generate_drift_log.py

K is never selected automatically. --silhouette-report additionally writes the Silhouette scores
for K in 2..20 as a diagnostic, it does not change the clustering. If sublogs already exist in
the output directory their partition is compared to the new one (adjusted Rand index).

The decoder step (step4_trace_pool_generation.py) needs per-cluster training folders in the
TF-decoder input format. See README.md.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import step2_resource_calendars as s2  # noqa: E402
import step3_trace_clustering as s3  # noqa: E402
from pipeline import load_config  # noqa: E402


def build_case_attributes(df, specs):
    names = []
    for spec in specs or []:
        src, name, tr = spec["source"], spec["name"], spec.get("transform", "none")
        if src not in df.columns:
            raise SystemExit(f"case attribute {src!r} not in the raw log, columns are {list(df.columns)[:30]}...")
        col = df[src]
        if tr == "category":
            df[name] = pd.factorize(col)[0].astype(float)
        else:
            vals = pd.to_numeric(col, errors="coerce")
            if vals.isna().all():
                raise SystemExit(f"case attribute {src!r} has no numeric values")
            df[name] = np.log1p(vals) if tr == "log1p" else vals
        # one value per case, the first non-null one
        df[name] = df.groupby("case_id")[name].transform(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
        df[name] = df[name].fillna(df[name].median())
        names.append(name)
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=None, help="sublog directory (default sublogs_dir of the config)")
    ap.add_argument("--silhouette-report", action="store_true")
    ap.add_argument("--skip-calendars", action="store_true")
    args = ap.parse_args()

    os.chdir(ROOT)
    cfg = load_config(args.config)
    dataset, k = cfg["dataset"], cfg["n_clusters"]
    cl = cfg.get("clustering") or {}
    out_dir = args.out_dir or cfg["sublogs_dir"]

    if not os.path.exists(cfg["raw_log"]):
        raise SystemExit(f"raw log {cfg['raw_log']} not found. See the config for where to obtain it.")
    df = s2.load_and_prepare_log(cfg["raw_log"], cfg.get("columns"))
    if "lifecycle" in df.columns and cl.get("keep_lifecycle", "all") == "complete":
        before = len(df)
        df = df[df["lifecycle"].astype(str).str.upper() == "COMPLETE"].copy()
        print(f"kept {len(df)}/{before} COMPLETE events for clustering")
    if "resource" not in df.columns:
        df["resource"] = "UNKNOWN"
    attr_names = build_case_attributes(df, cl.get("case_attributes"))
    keep = ["case_id", "activity", "resource", "timestamp"] + attr_names
    df = df[keep].sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    print(f"{df['case_id'].nunique()} cases, {len(df)} events, case attributes {attr_names}")

    feats = s3.extract_all_features(df, extra_case_cols=tuple(attr_names), progress_every=0)
    case_ids = list(feats.keys())
    fv = [feats[c] for c in case_ids]

    if args.silhouette_report:
        res = s3.cluster_pipeline(fv, dataset, seed=cl.get("seed", 42), verbose=False)
        rep = pd.DataFrame({"k": res["k_range"], "silhouette": res["sil_scores"]})
        os.makedirs("outputs", exist_ok=True)
        rep.to_csv(f"outputs/{dataset}_silhouette.csv", index=False)
        print(f"Silhouette maximum at K={res['best_k']} (diagnostic only, K={k} is used)")
        print(rep.round(3).to_string(index=False))

    res = s3.cluster_fixed_k(fv, dataset, k, target_variance=cl.get("target_variance", 0.80),
                             seed=cl.get("seed", 42))
    labels = pd.Series(res["labels"], index=case_ids)
    order = labels.value_counts().index.tolist()  # C1 is the largest cluster
    concept = labels.map({raw: f"C{i + 1}" for i, raw in enumerate(order)})

    existing = {}
    for i in range(1, k + 1):
        pth = os.path.join(out_dir, f"{dataset}_sublog_C{i}.csv")
        if os.path.exists(pth):
            ids = pd.read_csv(pth, usecols=["case_id"])["case_id"].astype(str).unique()
            existing.update({c: f"C{i}" for c in ids})
    if existing:
        from sklearn.metrics import adjusted_rand_score
        common = [c for c in case_ids if c in existing]
        ari = adjusted_rand_score([existing[c] for c in common], concept.loc[common].tolist())
        print(f"agreement with existing sublogs in {out_dir}, {len(common)} cases, adjusted Rand index {ari:.4f}")

    summary = s3.export_sublogs(df, concept.rename("concept"), dataset, out_dir=out_dir)
    print(summary.to_string(index=False))

    if not args.skip_calendars:
        s2.load_or_discover_calendars(cfg["raw_log"], cfg["calendars_cache"], cfg.get("columns"),
                                      **(cfg.get("calendar_params") or {}))
    print("\nNext, train the TF-decoder per cluster (step4_trace_pool_generation.py) to obtain "
          f"generated_{dataset}_rank*.csv, then run generate_drift_log.py.")


if __name__ == "__main__":
    main()
