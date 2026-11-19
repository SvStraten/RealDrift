"""Per-task arrival models. AT-KDE (global, weekday and intraday decomposition) is the model
used in the paper. A flat KDE over inter-arrival times is kept only as an explicit fallback."""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


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
                                 extend_days=365, verbose=False, seed=None):
    """Return (sampler, reason). sampler is None and reason names the cause if AT-KDE is unusable."""
    try:
        from atkde_adapter import ATKDESampler
    except Exception as e:
        return None, f"AT-KDE not importable ({e!r})"
    try:
        ts = sorted(pd.Timestamp(t) for t in pd.Series(arrival_times))
        sampler = ATKDESampler(ts, kwargs=kde_kwargs, horizon_days=horizon_days,
                                extend_days=extend_days, verbose=verbose, seed=seed)
        return sampler, None
    except Exception as e:
        return None, f"AT-KDE construction failed ({e!r})"


class _GapSource:
    """Uniform interface over AT-KDE and the flat-KDE fallback.

    strict=True raises instead of falling back, so a run can never silently use a different
    arrival model than the one reported. describe() is written to the run manifest.
    """

    def __init__(self, arrival_times, use_atkde=False, atkde_kwargs=None, seed=None, strict=False):
        self.flat_kde = fit_flat_kde(arrival_times)
        self.requested_atkde = bool(use_atkde)
        self.strict = strict
        self.fallback_reason = None
        self.n_atkde_samples = 0
        self.n_flat_samples = 0
        self.atkde = None
        if use_atkde:
            kwargs = dict(atkde_kwargs or {})
            self.atkde, self.fallback_reason = try_construct_atkde_sampler(
                arrival_times, seed=seed, **kwargs)
            if self.atkde is None:
                if strict:
                    raise RuntimeError(f"AT-KDE requested but unavailable: {self.fallback_reason}")
                warnings.warn(f"falling back to flat KDE: {self.fallback_reason}", stacklevel=2)

    @property
    def used_atkde_ever(self):
        return self.n_atkde_samples > 0

    def sample(self, current_time, rng):
        if self.atkde is not None:
            try:
                g = max(float(self.atkde.sample_gap(current_time)), 60.0)
                self.n_atkde_samples += 1
                return g
            except Exception as e:
                self.fallback_reason = f"AT-KDE failed mid-run at {current_time} ({e!r})"
                if self.strict:
                    raise RuntimeError(self.fallback_reason) from e
                warnings.warn(f"{self.fallback_reason}, using flat KDE for the rest of this task")
                self.atkde = None
        self.n_flat_samples += 1
        return sample_gap(self.flat_kde, rng)

    def describe(self):
        return {"requested_atkde": self.requested_atkde,
                "n_atkde_samples": self.n_atkde_samples,
                "n_flat_samples": self.n_flat_samples,
                "fallback_reason": self.fallback_reason}


def make_gap_source(arrival_times, use_atkde=False, atkde_kwargs=None, seed=None, strict=False):
    return _GapSource(arrival_times, use_atkde=use_atkde, atkde_kwargs=atkde_kwargs,
                      seed=seed, strict=strict)
