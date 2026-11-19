import pandas as pd
import pytest

from io_utils import parse_timestamps
from evaluation.scoring import load_ground_truth, score_detections, resolve_tolerance


def ts(s):
    return pd.Timestamp(s, tz="UTC")


def gt_frame(rows):
    return load_ground_truth(pd.DataFrame(rows, columns=["start_ts", "end_ts", "type"]))


def test_parse_timestamps_keeps_mixed_fractional_seconds():
    s = pd.Series(["2011-10-01 06:11:08.866000+00:00", "2011-10-02 14:53:33+00:00"])
    out = parse_timestamps(s)
    assert out.notna().all() and out.iloc[1] == ts("2011-10-02 14:53:33")


def test_parse_timestamps_raises_on_garbage():
    with pytest.raises(ValueError):
        parse_timestamps(pd.Series(["2011-10-01 06:11:08", "not a date"]))


def test_perfect_detection():
    gt = gt_frame([("2020-01-10", "2020-01-10", "sudden"), ("2020-02-10", "2020-02-12", "gradual")])
    s = score_detections([ts("2020-01-10 01:00"), ts("2020-02-11")], gt, pd.Timedelta(days=1))
    assert (s.tp, s.fp, s.fn, s.f1) == (2, 0, 0, 1.0)


def test_gradual_interval_and_tolerance_edge():
    gt = gt_frame([("2020-02-10", "2020-02-20", "gradual")])
    inside = score_detections([ts("2020-02-15")], gt, pd.Timedelta(0))
    edge = score_detections([ts("2020-02-21")], gt, pd.Timedelta(days=1))
    outside = score_detections([ts("2020-02-21 00:01")], gt, pd.Timedelta(days=1))
    assert inside.tp == 1 and edge.tp == 1 and outside.tp == 0


def test_matching_is_one_to_one_and_optimal():
    # greedy matching in ground truth order would give the single detection to the first drift
    # and miss nothing here, but two detections and overlapping tolerance windows need the optimum
    gt = gt_frame([("2020-01-10", "2020-01-10", "sudden"), ("2020-01-12", "2020-01-12", "sudden")])
    s = score_detections([ts("2020-01-11"), ts("2020-01-09")], gt, pd.Timedelta(days=1.5))
    assert s.tp == 2 and s.fp == 0 and s.fn == 0


def test_no_double_matching_and_false_positives():
    gt = gt_frame([("2020-01-10", "2020-01-10", "sudden")])
    s = score_detections([ts("2020-01-10"), ts("2020-01-10 02:00")], gt, pd.Timedelta(days=1))
    assert (s.tp, s.fp, s.fn) == (1, 1, 0) and s.precision == 0.5 and s.recall == 1.0


def test_empty_detections_and_conventions():
    gt = gt_frame([("2020-01-10", "2020-01-10", "sudden")])
    s = score_detections([], gt, pd.Timedelta(days=1))
    assert (s.precision, s.recall, s.f1, s.fn) == (0.0, 0.0, 0.0, 1)


def test_tolerance_in_windows_needs_resolution():
    assert resolve_tolerance({"unit": "windows", "value": 2}, 3600) == pd.Timedelta(hours=2)
    with pytest.raises(ValueError):
        resolve_tolerance({"unit": "windows", "value": 1}, None)


def test_span_fraction_tolerance():
    span = 1000 * 86400.0
    assert resolve_tolerance({"unit": "span_fraction", "value": 0.02}, None, span) == pd.Timedelta(days=20)
    with pytest.raises(ValueError):
        resolve_tolerance({"unit": "span_fraction", "value": 0.02})
