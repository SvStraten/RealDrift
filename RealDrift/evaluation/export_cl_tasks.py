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
