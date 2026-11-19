import os

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT

SUBLOG = os.path.join(ROOT, "data", "sublogs", "bpic12_sublog_C5.csv")


@pytest.mark.skipif(not os.path.exists(SUBLOG), reason="BPIC2012 sublogs not available")
def test_atkde_sampling_depends_on_seed_only():
    from io_utils import load_pool
    from step5_arrival_time_modeling import make_gap_source
    _, arr = load_pool(SUBLOG, verbose=False)

    def run(seed):
        gs = make_gap_source(arr, use_atkde=True, seed=seed, strict=True)
        rng, t, out = np.random.default_rng(0), pd.Timestamp("2012-01-01", tz="UTC"), []
        for _ in range(30):
            t = t + pd.to_timedelta(gs.sample(t, rng), unit="s")
            out.append(t)
        assert gs.describe()["n_flat_samples"] == 0
        return out

    a, b, c = run(0), run(0), run(1)
    assert a == b and a != c
