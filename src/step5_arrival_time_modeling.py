import warnings

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


#this is the fallback kde, not the one we are using in the paper 
def fit_flat_kde(arrival_times):
    ts = pd.to_datetime(pd.Series(arrival_times)).sort_values()
    gaps = ts.diff().dt.total_seconds().dropna().to_numpy()
    gaps = gaps[gaps > 0]
    if len(gaps) < 3:
        warnings.warn("fewer than 3 positive inter-arrival gaps -- using a generic "
                       "1-hour-median exponential fallback; check your data.")
        gaps = np.random.default_rng(0).exponential(3600, size=200)
    return gaussian_kde(gaps)


def sample_gap(kde, rng, floor_s=60.0):
    g = float(kde.resample(1, seed=np.random.RandomState(rng.integers(0, 2**31 - 1)))[0, 0])
    return max(g, floor_s)


def try_construct_atkde_sampler(arrival_times, kde_kwargs=None, horizon_days=730,
                                 extend_days=365, verbose=False):
    try:
        from atkde_adapter import ATKDESampler
    except Exception as e:
        print(f"[step5_arrival_time_modeling] atkde_adapter/AT-KDE not importable ({e!r}); "
              f"falling back to flat KDE.")
        return None
    try:
        ts = sorted(pd.Timestamp(t) for t in pd.Series(arrival_times))
        sampler = ATKDESampler(ts, kwargs=kde_kwargs, horizon_days=horizon_days,
                                extend_days=extend_days, verbose=verbose)
        print(f"[step5_arrival_time_modeling] ATKDESampler constructed for {len(ts)} arrivals "
              f"spanning {ts[0]} .. {ts[-1]} (horizon_days={horizon_days}).")
        return sampler
    except Exception as e:
        print(f"[step5_arrival_time_modeling] ATKDESampler construction failed ({e!r}); "
              f"falling back to flat KDE.")
        return None


class _GapSource:
    def __init__(self, arrival_times, use_atkde=False, atkde_kwargs=None):
        self.flat_kde = fit_flat_kde(arrival_times)
        self.atkde = (try_construct_atkde_sampler(arrival_times, **(atkde_kwargs or {}))
                      if use_atkde else None)
        self.used_atkde_ever = False

    def sample(self, current_time, rng):
        if self.atkde is not None:
            try:
                g = max(float(self.atkde.sample_gap(current_time)), 60.0)
                self.used_atkde_ever = True
                return g
            except Exception as e:
                print(f"[step5_arrival_time_modeling] ATKDESampler.sample_gap failed ({e!r}) "
                      f"mid-run (current_time={current_time}); falling back to flat KDE for "
                      f"the rest of this concept.")
                self.atkde = None
        return sample_gap(self.flat_kde, rng)


def make_gap_source(arrival_times, use_atkde=False, atkde_kwargs=None):
    return _GapSource(arrival_times, use_atkde=use_atkde, atkde_kwargs=atkde_kwargs)
