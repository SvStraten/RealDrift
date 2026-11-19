from evaluation.detector_pelt import pool_change_points


def test_pool_merges_nearby_change_points_and_filters_by_support():
    cps = {"a": [10, 30], "b": [11], "c": [10, 55]}
    assert pool_change_points(cps, merge_windows=1, min_features=1) == [10, 30, 55]
    assert pool_change_points(cps, merge_windows=1, min_features=2) == [10]
    assert pool_change_points(cps, merge_windows=1, min_features=3) == [10]
    assert pool_change_points({"a": []}, 1, 1) == []


def test_detectors_register():
    from evaluation import detectors as D
    import evaluation.detector_pelt, evaluation.detector_vc  # noqa: F401
    assert {"pelt", "vc"} <= set(D.REGISTRY)
