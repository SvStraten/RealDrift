"""Export a composed log with task labels for continual learning experiments.

The task of a case is the cluster it was generated from (C1A -> task 1, C1B -> task 1). The
export has one row per event in time order with the columns case_id, activity, resource, timestamp,
instance (C1A, C1B, ...), task (1..K) and instance_index (position in the sequence). The
transitions file lists the task boundaries. The exact input format of the continual learning
method (COMPASS) has to be matched on top of this file.

    python -m evaluation.export_cl_tasks outputs/streams/eval/bpic12_recurrent_seed0 --out cl_bpic12_seed0.csv
"""
from __future__ import annotations

import argparse
import re

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix", help="path prefix of a composed run, without the _drift_log.csv suffix")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    log = pd.read_csv(a.prefix + "_drift_log.csv")
    log["timestamp"] = pd.to_datetime(log["time:timestamp"], utc=True, format="ISO8601")
    seq = pd.read_csv(a.prefix + "_transitions.csv")
    order = list(dict.fromkeys(list(seq["from"]) + list(seq["to"])))
    inst = log["case:concept_label"]
    out = pd.DataFrame({
        "case_id": log["case:concept:name"], "activity": log["concept:name"],
        "resource": log["org:resource"], "timestamp": log["timestamp"].astype(str),
        "instance": inst, "task": inst.map(lambda s: int(re.match(r"C(\d+)", s).group(1))),
        "instance_index": inst.map({l: i for i, l in enumerate(order)})})
    out.sort_values("timestamp").to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(out)} events, {out['instance'].nunique()} instances)")


if __name__ == "__main__":
    main()
