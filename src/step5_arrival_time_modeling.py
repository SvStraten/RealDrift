"""
step5_arrival_time_modeling.py

Arrival time modeling (subsection 4.4, arrival-model half; the global
conditions / feasibility half lives in step6_drift_composer.py).

Per task, fits a generative model of inter-arrival gaps from that task's
real arrival timestamps. Two backends:
  - flat_kde: a single gaussian KDE over inter-arrival gaps. The default
    used throughout this repo's generate_drift_log.py runs.
  - AT-KDE (Kirchdorfer et al.): the global/weekday/time-of-day
    decomposition, via atkde_adapter.py, which wraps
    github.com/konradoezdemir/AT-KDE rather than reimplementing it. Used
    when that repo is cloned and importable (use_atkde=True in
    make_gap_source, or the use_atkde config flag in
    generate_drift_log.py).
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def fit_flat_kde(arrival_times):
    """Single gaussian KDE over inter-arrival gaps (seconds) between this
    concept's real case arrivals. No global/weekday/time-of-day
    structure, just one distribution over gap length.

    Gaps are computed with `.dt.total_seconds()` rather than a manual
    `ts.astype("int64") / 1e9` conversion, since the latter assumes
    nanosecond datetime64 resolution. Pandas 2.x's
    `pd.to_datetime(..., utc=True)` can return microsecond resolution
    instead, which would silently understate every gap by 1000x with
    `/ 1e9`. `.dt.total_seconds()` is resolution-independent.
    """
    ts = pd.to_datetime(pd.Series(arrival_times)).sort_values()
    gaps = ts.diff().dt.total_seconds().dropna().to_numpy()
    gaps = gaps[gaps > 0]
    if len(gaps) < 3:
        warnings.warn("fewer than 3 positive inter-arrival gaps -- using a generic "
                       "1-hour-median exponential fallback; check your data.")
        gaps = np.random.default_rng(0).exponential(3600, size=200)
    return gaussian_kde(gaps)


def sample_gap(kde, rng, floor_s=60.0):
    """Draws one inter-arrival gap (seconds) from a fitted KDE, floored so
    a non-positive or negligibly small gap is never sampled."""
    g = float(kde.resample(1, seed=np.random.RandomState(rng.integers(0, 2**31 - 1)))[0, 0])
    return max(g, floor_s)


def try_construct_atkde_sampler(arrival_times, kde_kwargs=None, horizon_days=730,
                                 extend_days=365, verbose=False):
    """Best-effort import and construction of ATKDESampler
    (atkde_adapter.py wrapping github.com/konradoezdemir/AT-KDE). Returns
    a sampler exposing `.sample_gap(current_time) -> seconds`, or None if
    the import or construction fails, in which case callers fall back to
    fit_flat_kde/sample_gap.

    kde_kwargs is forwarded as atkde_adapter's `kwargs` dict (e.g.
    {'lower':.., 'upper':..} to fix a numeric domain), passed as one dict
    argument rather than **-expanded, matching the real constructor.

    Construction itself is cheap (builds a KDEIATGenerator); the
    expensive part (segmentation and per-cluster bandwidth optimization)
    runs on the first sample_gap() call, which lazily generates the
    initial `horizon_days`-long window anchored at whatever current_time
    is first requested. This function does not call sample_gap() itself,
    to avoid triggering that cost for every concept even when a run only
    uses a few of them.

    Needs roughly 1-2k+ arrivals spanning multiple months to work
    reliably; the underlying segmentation/outlier-detection step can
    raise on very small or short arrival series, in which case this
    falls back to flat KDE and prints why.
    """
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
    """Uniform interface so composition code does not need to know
    whether a concept is using ATKDESampler or the flat-KDE fallback.
    sample() takes the composer's current absolute simulated time (a
    pd.Timestamp): ATKDESampler uses it to query the next arrival from
    its own pre-generated queue; the flat-KDE path ignores it and draws a
    random gap.

    used_atkde_ever is True if ATKDESampler was successfully used for at
    least one sample() call during this gap source's lifetime. This is
    distinct from checking `self.atkde is not None` after the fact, since
    a mid-run sample_gap() failure resets self.atkde to None.
    """
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
    """atkde_kwargs, if given, is forwarded to try_construct_atkde_sampler,
    e.g. {'horizon_days': 120, 'extend_days': 60, 'kde_kwargs': {...}}."""
    return _GapSource(arrival_times, use_atkde=use_atkde, atkde_kwargs=atkde_kwargs)
