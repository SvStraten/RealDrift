from __future__ import annotations

import logging

import pandas as pd

from source.iat_approaches.kde import KDEIATGenerator


class ATKDESampler:
    def __init__(self, train_arrival_times, kwargs: dict | None = None,
                 horizon_days: int = 730, extend_days: int = 365,
                 verbose: bool = False):
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

    def _extend(self, from_time: pd.Timestamp, days: int) -> None:
        end = from_time + pd.Timedelta(days=days)
        new_arrivals = self._generator.generate_arrivals(from_time, end)
        new_arrivals = sorted(pd.Timestamp(a) for a in new_arrivals)
        
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
        current_time = pd.Timestamp(current_time)

        if self._horizon_end is None:
            self._extend(current_time, self._horizon_days)

        if self._queue:
            queue_tz = self._queue[0].tzinfo
            if current_time.tzinfo is None and queue_tz is not None:
                current_time = current_time.tz_localize(queue_tz)
            elif current_time.tzinfo is not None and queue_tz is None:
                current_time = current_time.tz_localize(None)

        EPS = 1e-6  

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
