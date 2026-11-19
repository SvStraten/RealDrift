import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

ACTS = {1: ["A", "B", "C"], 2: ["A", "D", "E", "C"], 3: ["F", "G"]}
RES = {1: ["r1", "r2"], 2: ["r2", "r3"], 3: ["r1", "r3"]}


def _case_rows(case_id, rank, start, rng):
    n = len(ACTS[rank]) + int(rng.integers(0, 2))
    t = start
    rows = []
    for k in range(n):
        act = ACTS[rank][k % len(ACTS[rank])]
        rows.append((case_id, act, RES[rank][k % 2], t))
        t = t + pd.Timedelta(minutes=int(rng.integers(5, 90)))
    return rows


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    """A small synthetic dataset with 3 clusters and A/B pools, composed without AT-KDE."""
    root = tmp_path_factory.mktemp("synth")
    sub, pools = root / "sublogs", root / "pools"
    sub.mkdir()
    pools.mkdir()
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2020-01-06 08:00", tz="UTC")
    for rank in (1, 2, 3):
        rows, t = [], base
        for i in range(50):
            t = t + pd.Timedelta(minutes=int(rng.integers(20, 240)))
            rows += _case_rows(f"s{rank}_{i}", rank, t, rng)
        pd.DataFrame(rows, columns=["case_id", "activity", "resource", "timestamp"]).to_csv(
            sub / f"toy_sublog_C{rank}.csv", index=False)
        for letter in "AB":
            rows = []
            for i in range(50):
                rows += _case_rows(f"g{i}", rank, base + pd.Timedelta(days=i), rng)
            df = pd.DataFrame(rows, columns=["case:concept:name", "concept:name", "org:resource", "time:timestamp"])
            df["time:timestamp"] = df["time:timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            df.to_csv(pools / f"generated_toy_rank{rank}{letter}.csv", sep=";", index=False)
    cfg = {"dataset": "toy", "n_clusters": 3, "t0": "auto", "sublogs_dir": str(sub),
           "concept_pools_dir": str(pools), "use_atkde": False, "unconstrained_resources": [],
           "composer": {"sa_iterations": 200, "gradual_window": 10}}
    path = root / "toy.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return {"config": str(path), "root": root}


@pytest.fixture()
def toy_cfg(synth):
    import pipeline
    return pipeline.load_config(synth["config"])
