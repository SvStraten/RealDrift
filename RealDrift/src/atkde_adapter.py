from __future__ import annotations

import logging
import os
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

_ATKDE_ROOT = os.environ.get(
    "ATKDE_ROOT", str(Path(__file__).resolve().parents[1] / "external" / "AT-KDE"))
if _ATKDE_ROOT not in sys.path:
    sys.path.insert(0, _ATKDE_ROOT)

from source.iat_approaches.kde import KDEIATGenerator  # noqa: E402


class _SeededGlobalRNG:
    """Context manager that seeds numpy and random globally and restores their state on exit."""

    def __init__(self, seed):
        self.seed = None if seed is None else int(seed) % (2 ** 32)

    def __enter__(self):
        if self.seed is not None:
            self._np_state = np.random.get_state()
            self._py_state = random.getstate()
            np.random.seed(self.seed)
            random.seed(self.seed)
        return self

    def __exit__(self, *exc):
        if self.seed is not None:
            np.random.set_state(self._np_state)
            random.setstate(self._py_state)
        return False


class ATKDESampler:
    def __init__(self, train_arrival_times, kwargs: dict | None = None,
                 horizon_days: int = 730, extend_days: int = 365,
                 verbose: bool = False, seed: int | None = None):
        self.logger = logging.getLogger(__name__)
        if not verbose:
            logging.getLogger("kde.py").setLevel(logging.WARNING)

        self.train = sorted(pd.Timestamp(t) for t in train_arrival_times)
        self.kwargs = kwargs or {}
        self.extend_days = extend_days
        self.seed = seed
        self._generator = KDEIATGenerator(train_arrival_times=self.train, kwargs=self.kwargs)

        self._queue: deque = deque()
        self._horizon_end: pd.Timestamp | None = None
        self._horizon_days = horizon_days
        self._n_extends = 0

    def _extend(self, from_time: pd.Timestamp, days: int) -> None:
        end = from_time + pd.Timedelta(days=days)
        call_seed = None if self.seed is None else self.seed * 1_000_003 + self._n_extends
        with _SeededGlobalRNG(call_seed):
            new_arrivals = self._generator.generate_arrivals(from_time, end)
        self._n_extends += 1
        new_arrivals = sorted(pd.Timestamp(a) for a in new_arrivals)

        if new_arrivals:
            tz = new_arrivals[0].tzinfo
            if end.tzinfo is None and tz is not None:
                end = end.tz_localize(tz)
            if self._queue and self._queue[0].tzinfo != tz:
                self._queue = deque(a.tz_localize(tz) if a.tzinfo is None else a.tz_convert(tz)
                                    for a in self._queue)

        new_arrivals = [a for a in new_arrivals if not self._queue or a > self._queue[-1]]
        self._queue.extend(new_arrivals)
        self._horizon_end = end

    def sample_gap(self, current_time: pd.Timestamp, max_extends: int = 20) -> float:
        """Seconds until the next queued arrival strictly after current_time."""
        current_time = pd.Timestamp(current_time)

        if self._horizon_end is None:
            self._extend(current_time, self._horizon_days)

        if self._queue:
            queue_tz = self._queue[0].tzinfo
            if current_time.tzinfo is None and queue_tz is not None:
                current_time = current_time.tz_localize(queue_tz)
            elif current_time.tzinfo is not None and queue_tz is None:
                current_time = current_time.tz_localize(None)

        EPS = 1e-6  # float rounding in the caller must not leave an arrival stuck in the queue

        extends = 0
        while True:
            while self._queue and (self._queue[0] - current_time).total_seconds() <= EPS:
                self._queue.popleft()
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
