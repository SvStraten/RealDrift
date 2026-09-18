"""
atkde_adapter.py

Adapter around the AT-KDE implementation from
https://github.com/konradoezdemir/AT-KDE (Kirchdorfer & Oezdemir et al.),
not a reimplementation.

Why an adapter is needed: the repo's own entry point,
KDEIATGenerator.generate_arrivals(start_time, end_time), generates a
batch of arrivals for a fixed window in one call (internally re-running
segmentation/clustering and per-cluster bandwidth optimization each
time). The drift composer instead needs an incremental
sample_gap(current_time) -> seconds-until-next-arrival, called
repeatedly as it interleaves multiple concepts' arrivals into one
stream. This adapter bridges the two: it calls the real generator once
for a generous horizon, queues the resulting arrivals, and pops them
off in order, extending the queue only if the composer runs past the
pre-generated horizon.

Setup:
    git clone https://github.com/konradoezdemir/AT-KDE.git
    pip install KDEpy pm4py tqdm scikit-learn scipy pandas numpy

    torch, tensorflow, prophet, chronos-forecasting, xgboost, rpy2, etc.
    from the repo's own requirements.txt are not needed; those back
    other baseline methods (LSTM/XGBoost/Prophet/Chronos/NPP) that this
    adapter does not call. The 'at_kde' method path only touches
    source/arrival_segmentation.py, kde_core/kde_simulator.py, and
    utils/helper.py, whose only third-party dependency beyond the stack
    above is KDEpy.

    NumPy 2.0 removed `np.infty`; the AT-KDE repo uses it in
    source/iat_approaches/kde.py and diagnostics/eval_event_logs.py.
    Replace both occurrences with `np.inf` after cloning if running on
    NumPy 2.x.

Usage:
    import sys
    sys.path.insert(0, '/path/to/cloned/AT-KDE')
    from atkde_adapter import ATKDESampler

    kde1 = ATKDESampler(real1['start'].tolist(), horizon_days=730)
    ...
    t_target += kde1.sample_gap(t0 + pd.to_timedelta(t_target, unit='s'))
"""

from __future__ import annotations

import logging

import pandas as pd

from source.iat_approaches.kde import KDEIATGenerator


class ATKDESampler:
    """Incremental sample_gap(current_time) wrapper around the real
    KDEIATGenerator.generate_arrivals(start_time, end_time).
    """

    def __init__(self, train_arrival_times, kwargs: dict | None = None,
                 horizon_days: int = 730, extend_days: int = 365,
                 verbose: bool = False):
        """
        train_arrival_times: list/array/Series of arrival timestamps for
            ONE task/cluster (e.g. real1['start']) -- matches the repo's own
            expected input to KDEIATGenerator (a list of pd.Timestamp).
        kwargs: forwarded to KDEIATGenerator (e.g. {'lower':..,'upper':..}
            to fix a numeric domain; usually leave empty for event-log data).
        horizon_days: length of the initial generated window, starting at
            the last training arrival. Set this to comfortably exceed the
            longest gap the composer might need to bridge for this concept
            (e.g. concept pool size / expected throughput).
        extend_days: length of each further window if the composer's
            current_time runs past the pre-generated horizon.
        """
        self.logger = logging.getLogger(__name__)
        if not verbose:
            logging.getLogger("kde.py").setLevel(logging.WARNING)

        self.train = sorted(pd.Timestamp(t) for t in train_arrival_times)
        self.kwargs = kwargs or {}
        self.extend_days = extend_days
        self._generator = KDEIATGenerator(train_arrival_times=self.train, kwargs=self.kwargs)

        self._queue: list[pd.Timestamp] = []
        self._horizon_end: pd.Timestamp | None = None
        self._horizon_days = horizon_days
        # Generation is lazy (see sample_gap): the first call anchors the
        # initial window to whatever current_time is actually asked for,
        # rather than assuming it will be near train[-1]. In the composer,
        # a concept's sampler can first be queried well after construction
        # (once that concept becomes 'active' in pick_source()), and at a
        # calendar time that has little to do with train[-1] specifically.

    def _extend(self, from_time: pd.Timestamp, days: int) -> None:
        end = from_time + pd.Timedelta(days=days)
        new_arrivals = self._generator.generate_arrivals(from_time, end)
        new_arrivals = sorted(pd.Timestamp(a) for a in new_arrivals)

        # the repo's KDE core may return tz-aware timestamps regardless of
        # whether from_time/end were tz-naive; once we've seen a real
        # result, normalize everything (queue, horizon_end) to that tzinfo
        # so later comparisons never mix aware/naive.
        if new_arrivals:
            tz = new_arrivals[0].tzinfo
            if end.tzinfo is None and tz is not None:
                end = end.tz_localize(tz)
            if self._queue and self._queue[0].tzinfo != tz:
                self._queue = [a.tz_localize(tz) if a.tzinfo is None else a.tz_convert(tz)
                               for a in self._queue]

        new_arrivals = [a for a in new_arrivals if not self._queue or a > self._queue[-1]]
        self._queue.extend(new_arrivals)
        self._horizon_end = end

    def sample_gap(self, current_time: pd.Timestamp, max_extends: int = 20) -> float:
        """Seconds until the next queued arrival strictly after
        `current_time`, extending the generated horizon if necessary."""
        current_time = pd.Timestamp(current_time)

        if self._horizon_end is None:
            # first-ever call: anchor the initial window to current_time
            # itself, not to train[-1] (see __init__ comment). Do this
            # before any tz handling -- generate_arrivals may return
            # tz-aware timestamps even if the request was tz-naive (this
            # repo's KDE core localizes internally), so we only know for
            # sure what tzinfo to match after the first real generation.
            self._extend(current_time, self._horizon_days)

        if self._queue:
            queue_tz = self._queue[0].tzinfo
            if current_time.tzinfo is None and queue_tz is not None:
                current_time = current_time.tz_localize(queue_tz)
            elif current_time.tzinfo is not None and queue_tz is None:
                current_time = current_time.tz_localize(None)

        # Callers (the composer) reconstruct current_time as
        # t0 + pd.to_timedelta(cumulative_float_seconds, unit='s'), which
        # accumulates float rounding error. Without slack, current_time can
        # land a fraction of a microsecond BEFORE a queued arrival on every
        # call, so the item never satisfies '<= current_time' and never
        # gets popped -- sample_gap would then return a near-zero gap
        # forever instead of advancing. Treat anything within EPS as
        # already arrived.
        EPS = 1e-6  # seconds

        extends = 0
        while True:
            while self._queue and (self._queue[0] - current_time).total_seconds() <= EPS:
                self._queue.pop(0)
            if self._queue:
                return (self._queue[0] - current_time).total_seconds()
            if extends >= max_extends:
                raise RuntimeError(
                    f"ATKDESampler: exhausted {max_extends} extensions without "
                    f"producing an arrival after {current_time}. The underlying "
                    f"process may have an unusually low rate near this point -- "
                    f"increase extend_days or check the training data."
                )
            self._extend(max(self._horizon_end, current_time), self.extend_days)
            extends += 1
