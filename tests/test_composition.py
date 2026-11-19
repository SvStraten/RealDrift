import pandas as pd
import pytest

import pipeline
from step6_drift_composer import _search_forward, feasible, commit, _event_specs
from validation import validate_composition


def weekday_calendar():
    """r1 works Monday to Friday 08:00 to 18:00, r2 and r3 have no calendar."""
    slots = {d * 24 + h for d in range(5) for h in range(8, 18)}
    return {"r1": slots}, set()


@pytest.mark.parametrize("tier,expected_n,expected_types", [
    ("sudden", 3, ["sudden"] * 2),
    ("gradual", 3, ["gradual"] * 2),
    ("recurrent", 6, ["sudden"] * 5),
    ("hard", 6, ["sudden", "gradual", "sudden", "gradual", "sudden"]),
])
def test_instance_plan(tier, expected_n, expected_types):
    plan, types = pipeline.instance_plan(tier, [1, 2, 3])
    assert len(plan) == expected_n
    assert types == expected_types


def test_hard_order_alternates_instances_of_one_task():
    plan, _ = pipeline.instance_plan("hard", [1, 2])
    assert plan == [(1, "A"), (1, "B"), (2, "A"), (2, "B")]


@pytest.mark.parametrize("tier", ["sudden", "gradual", "hard", "recurrent"])
def test_ground_truth_matches_final_log(toy_cfg, tier):
    res = pipeline.compose_run(toy_cfg, tier, 0, calendars=({}, set()), verbose=False)
    m = res["manifest"]
    assert m["ground_truth_check"]["ok"], m["ground_truth_check"]["problems"]
    log = res["log"]
    first = log.groupby("case:concept_label")["time:timestamp"].min()
    for tr in res["transitions"]:
        assert tr["start_ts"] <= tr["end_ts"]
        if tr["type"] == "sudden":
            assert tr["start_ts"] == tr["end_ts"]
        # the incoming instance really starts inside the recorded interval
        case_starts = [t for _, (c, t) in res["placed"].items() if c["concept"] == tr["to"]]
        assert min(case_starts) == tr["start_ts"] if tr["type"] == "sudden" else min(case_starts) >= tr["start_ts"]


def test_sudden_never_mixes_instances_before_boundary(toy_cfg):
    res = pipeline.compose_run(toy_cfg, "sudden", 0, calendars=({}, set()), skip_sa=True, verbose=False)
    placed = res["placed"]
    for tr in res["transitions"]:
        before = [t for _, (c, t) in placed.items() if c["concept"] == tr["to"] and t < tr["start_ts"]]
        assert not before


def test_gradual_interval_contains_both_instances(toy_cfg):
    res = pipeline.compose_run(toy_cfg, "gradual", 0, calendars=({}, set()), skip_sa=True, verbose=False)
    assert all(tr["n_overlap"] >= 2 for tr in res["transitions"])


def test_same_seed_is_reproducible_and_other_seed_differs(toy_cfg):
    a = pipeline.compose_run(toy_cfg, "hard", 3, calendars=({}, set()), verbose=False)["log"]
    b = pipeline.compose_run(toy_cfg, "hard", 3, calendars=({}, set()), verbose=False)["log"]
    c = pipeline.compose_run(toy_cfg, "hard", 4, calendars=({}, set()), verbose=False)["log"]
    pd.testing.assert_frame_equal(a, b)
    assert not a["time:timestamp"].equals(c["time:timestamp"])


def test_calendars_and_capacity_hold_in_final_log(toy_cfg):
    cal = weekday_calendar()
    res = pipeline.compose_run(toy_cfg, "hard", 1, calendars=cal, verbose=False)
    v = res["manifest"]["validation"]
    assert v["ok"], v
    assert v["calendar_violations"] == 0 and not v["resources_over_cap"]
    log = res["log"]
    r1 = pd.to_datetime(log[log["org:resource"] == "r1"]["time:timestamp"], utc=True)
    assert (r1.dt.weekday < 5).all() and r1.dt.hour.between(8, 17).all()


def test_bed_capacity(toy_cfg):
    toy_cfg["composer"]["max_beds"] = 3
    res = pipeline.compose_run(toy_cfg, "sudden", 0, calendars=({}, set()), verbose=False)
    assert res["manifest"]["validation"]["max_concurrent_cases"] <= 3
    assert res["manifest"]["validation"]["ok"]


def test_validator_detects_violations(toy_cfg):
    res = pipeline.compose_run(toy_cfg, "sudden", 0, calendars=({}, set()), skip_sa=True, verbose=False)
    placed = dict(res["placed"])
    # force two cases of different clusters onto the same time so one resource is double booked
    (k1, (c1, t1)), (k2, (c2, t2)) = [(k, v) for k, v in placed.items() if "r2" in v[0]["resources"]][:2]
    placed[k2] = (c2, t1 - pd.to_timedelta(c2["offsets"][c2["resources"].index("r2")], unit="s")
                  + pd.to_timedelta(c1["offsets"][c1["resources"].index("r2")], unit="s"))
    bad = validate_composition(placed, {}, set(), resource_cap=1)
    assert not bad["ok"] and "r2" in bad["resources_over_cap"]
    cal_bad = validate_composition(res["placed"], ({"r1": {0}}), set())
    assert cal_bad["calendar_violations"] > 0


def test_phase_search_is_equivalent_to_brute_force(toy_cfg):
    cal, pooled = weekday_calendar()
    res = pipeline.compose_run(toy_cfg, "sudden", 0, calendars=(cal, pooled), skip_sa=True, verbose=False)
    placed = dict(res["placed"])
    occ, store = {}, {}
    for cid, (c, t) in list(placed.items())[:25]:
        commit(c, t, occ, store, 15, None)
    start = pd.Timestamp("2020-01-08 07:30", tz="UTC")
    for cid, (c, _t) in list(placed.items())[25:45]:
        fast = _search_forward(c, start, occ, cal, pooled, 1, 15, 60, 21)
        t, slow = start, None
        for _ in range(int(21 * 24 * 4)):
            if feasible(c, t, occ, cal, pooled, 1, 15, 60, None, None)[0]:
                slow = t
                break
            t = t + pd.Timedelta(minutes=15)
        assert fast == slow


def test_unconstrained_resources_are_exempt(toy_cfg):
    toy_cfg["unconstrained_resources"] = ["r2", "r3"]
    res = pipeline.compose_run(toy_cfg, "sudden", 0, calendars=weekday_calendar(), skip_sa=True, verbose=False)
    assert res["manifest"]["validation"]["ok"]
    assert set(res["log"]["org:resource"]) >= {"r2", "r3"}   # still in the log


def test_ground_truth_validator_catches_a_misplaced_sudden_drift(toy_cfg):
    """The legacy composer logged a sudden drift at the START of the outgoing task. This must fail."""
    from validation import validate_ground_truth
    res = pipeline.compose_run(toy_cfg, "sudden", 0, calendars=({}, set()), verbose=False)
    assert validate_ground_truth(res["transitions"], res["placed"])["ok"]
    wrong = [dict(tr) for tr in res["transitions"]]
    first_start = min(t for _, (c, t) in res["placed"].items() if c["concept"] == wrong[0]["from"])
    wrong[0]["start_ts"] = wrong[0]["end_ts"] = first_start
    check = validate_ground_truth(wrong, res["placed"])
    assert not check["ok"] and check["problems"]
