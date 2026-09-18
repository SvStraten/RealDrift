"""
step1_attribute_classification.py

Attribute classification (subsection 4.1, first half). Classifies every
non-core column of an event log as case-level, global-level, or
event-level.

Pipeline:

1. Within-case invariance test. Wraps pix-framework's
   case_attribute_discovery module (pip install pix-framework, requires
   Python <3.12). Falls back to a direct reimplementation of the same
   method (confidence of each column's per-case mode, kept if above
   `confidence_threshold`) when pix-framework is not importable.

   This test alone only separates varying columns (event-level) from
   invariant columns (constant within a case). It does not separate case
   attributes from global conditions, since both are typically constant
   within a single case.

2. Case vs. global, for columns that passed test 1. See
   classify_case_vs_global / _global_vs_case_adjacency_test.

3. Event, for columns that failed test 1. See classify_event_columns.

On a log with no extra columns beyond what the case-attribute test
already finds (e.g. BPIC2012, whose only extra column, AMOUNT_REQ, is
case-level), tests 2 and 3 have nothing left to classify and return
empty lists.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Test 1: within-case invariance (case-or-global vs. event)
# --------------------------------------------------------------------------

def _case_attribute_confidence(event_log: pd.DataFrame, case_col: str, column: str) -> float:
    """Fraction of cases where this column's single most frequent value
    within that case accounts for the case's rows -- pix-framework's own
    documented method (confidence of the mode, per case, averaged)."""
    def _mode_confidence(s: pd.Series) -> float:
        counts = s.value_counts()
        return counts.iloc[0] / len(s) if len(s) else 0.0
    per_case = event_log.groupby(case_col)[column].apply(_mode_confidence)
    return per_case.mean()


def discover_invariant_columns(event_log: pd.DataFrame, case_col: str = "case_id",
                                avoid_columns=(), confidence_threshold: float = 0.9) -> list[str]:
    """Returns columns that are constant (up to confidence_threshold)
    within each case -- candidates for case OR global, not yet split.
    Tries the real pix-framework import first; falls back to a direct
    reimplementation of the same confidence-of-mode method if
    pix-framework isn't importable on this Python version (see module
    docstring)."""
    candidate_cols = [c for c in event_log.columns if c not in avoid_columns and c != case_col]

    try:
        from pix_framework.discovery.case_attribute.discovery import discover_case_attributes as _pix_discover
        from pix_framework.io.event_log import EventLogIDs

        log_ids = EventLogIDs(case=case_col)
        result = _pix_discover(event_log=event_log, log_ids=log_ids,
                                avoid_columns=list(avoid_columns), confidence_threshold=confidence_threshold)
        print("[step1] used the real pix-framework case_attribute_discovery module")
        return list(result)
    except ImportError:
        warnings.warn(
            "[step1] pix-framework not importable (it currently requires Python <3.12) -- "
            "falling back to a direct reimplementation of its own documented method "
            "(confidence of each column's per-case mode). Install pix-framework on a "
            "compatible Python to use the real package instead.",
            stacklevel=2,
        )
        return [col for col in candidate_cols
                if _case_attribute_confidence(event_log, case_col, col) >= confidence_threshold]


# --------------------------------------------------------------------------
# Test 2: case vs. global, for columns that passed test 1
# --------------------------------------------------------------------------

def _global_vs_case_adjacency_test(event_log: pd.DataFrame, case_col: str, time_col: str,
                                    column: str, n_shuffles: int = 200, seed: int = 0,
                                    rel_tol: float = 1e-6) -> tuple[str, float, float]:
    """Distinguishes a global attribute (a log-wide, case-independent
    state, e.g. beds available, constant across a single case but shared
    with whichever other cases are active at the same time and changing
    in synchronized blocks) from a case attribute (independently drawn
    per case, e.g. age).

    Sorts cases by their first event's timestamp, measures how often
    chronologically adjacent cases share the within-case-constant value,
    and compares that rate against the same statistic under n_shuffles
    random permutations of case order. A rate sufficiently above the
    shuffled baseline indicates a global attribute.
    """
    case_level = event_log.groupby(case_col).agg(**{
        "_val": (column, "first"), "_t0": (time_col, "min")}).sort_values("_t0")
    values = case_level["_val"].to_numpy()

    def _match_rate(v):
        if np.issubdtype(v.dtype, np.floating):
            denom = np.maximum(np.abs(v[1:]), np.abs(v[:-1]))
            denom = np.where(denom == 0, 1.0, denom)
            return float(np.mean(np.abs(v[1:] - v[:-1]) / denom < rel_tol))
        return float(np.mean(v[1:] == v[:-1]))

    actual_rate = _match_rate(values)
    rng = np.random.default_rng(seed)
    shuffled_rates = [_match_rate(rng.permutation(values)) for _ in range(n_shuffles)]
    baseline = float(np.mean(shuffled_rates))
    # Normalized excess over the headroom above chance, rather than a raw
    # ratio: a low-cardinality attribute has a high baseline match rate
    # from chance alone, so a raw actual/baseline ratio is poorly
    # calibrated there. Normalizing by the available headroom above
    # baseline corrects for this.
    headroom = max(1.0 - baseline, 1e-9)
    normalized_excess = (actual_rate - baseline) / headroom
    label = "global" if normalized_excess > 0.5 else "case"
    return label, actual_rate, baseline


def classify_case_vs_global(event_log: pd.DataFrame, case_col: str, time_col: str,
                             columns_to_classify: list[str], verbose: bool = True) -> dict:
    """{column: 'case' | 'global'} for columns that passed test 1."""
    result = {}
    for col in columns_to_classify:
        label, actual_rate, baseline = _global_vs_case_adjacency_test(event_log, case_col, time_col, col)
        result[col] = label
        if verbose:
            print(f"[step1] '{col}': constant within a case, chronologically-adjacent-case "
                  f"match rate={actual_rate:.2%} vs. shuffled baseline={baseline:.2%} "
                  f"-> classified as {label}")
    return result


# --------------------------------------------------------------------------
# Test 3: event, for columns that failed test 1
# --------------------------------------------------------------------------

def _reconstruct_event_hypothesis(event_log: pd.DataFrame, case_col: str, time_col: str, column: str) -> pd.Series:
    """pre-execution value = this case's own previous event's value."""
    ordered = event_log.sort_values([case_col, time_col])
    pre = ordered.groupby(case_col)[column].shift(1)
    return pre.reindex(event_log.index)


def _fit_update_rule_mse(pre: pd.Series, post: pd.Series) -> float:
    """Least-squares linear fit predicted_post = a*pre + b; returns
    in-sample MSE. Rows where pre is NaN (a case's first observation,
    nothing to predict from yet) are dropped from the fit."""
    mask = pre.notna() & post.notna()
    if mask.sum() < 2:
        return np.inf
    x, y = pre[mask].astype(float).values, post[mask].astype(float).values
    if np.std(x) == 0:
        a, b = 0.0, y.mean()
    else:
        a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    return float(np.mean((pred - y) ** 2))


def classify_event_columns(event_log: pd.DataFrame, case_col: str, time_col: str,
                            columns_to_classify: list[str], verbose: bool = True) -> dict:
    """{column: 'event'} for columns that failed test 1 (vary within a
    case). Implements the paper's reconstruction / update-rule-fit
    description (subsection 4.1) -- a column reaching this function has,
    by construction, already shown it varies within a case, so this fit
    mainly makes that explicit and inspectable rather than changing the
    outcome."""
    result = {}
    for col in columns_to_classify:
        if not pd.api.types.is_numeric_dtype(event_log[col]):
            if verbose:
                print(f"[step1] '{col}' is non-numeric -- classifying as event without "
                      f"the linear-fit test (the paper's method assumes a numeric attribute)")
            result[col] = "event"
            continue
        pre_event = _reconstruct_event_hypothesis(event_log, case_col, time_col, col)
        mse_event = _fit_update_rule_mse(pre_event, event_log[col])
        result[col] = "event"
        if verbose:
            print(f"[step1] '{col}': varies within a case (event-hypothesis fit MSE={mse_event:.4g}) "
                  f"-> classified as event")
    return result


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------

def classify_attributes(event_log: pd.DataFrame, case_col: str = "case_id", time_col: str = "timestamp",
                         core_columns=("activity", "resource"), confidence_threshold: float = 0.9,
                         verbose: bool = True) -> dict:
    """Runs all three tests. core_columns are the always-event-level
    columns every other step in this repo already assumes (activity,
    resource) and are never passed to any test. Returns
    {'case': [...], 'global': [...], 'event': [...]}."""
    avoid = list(core_columns) + [case_col, time_col]
    invariant_cols = discover_invariant_columns(event_log, case_col=case_col, avoid_columns=avoid,
                                                 confidence_threshold=confidence_threshold)
    varying_cols = [c for c in event_log.columns if c not in avoid and c not in invariant_cols]
    if verbose:
        print(f"[step1] within-case invariant (case-or-global candidates): {invariant_cols}")
        print(f"[step1] varies within a case (event candidates): {varying_cols}")

    case_cols, global_cols = [], []
    if invariant_cols:
        cg = classify_case_vs_global(event_log, case_col, time_col, invariant_cols, verbose=verbose)
        case_cols = [c for c, lbl in cg.items() if lbl == "case"]
        global_cols = [c for c, lbl in cg.items() if lbl == "global"]

    event_cols = []
    if varying_cols:
        ev = classify_event_columns(event_log, case_col, time_col, varying_cols, verbose=verbose)
        event_cols = [c for c, lbl in ev.items() if lbl == "event"]

    if not invariant_cols and not varying_cols and verbose:
        print("[step1] no extra columns to classify beyond the core/case/time columns")

    return {"case": case_cols, "global": global_cols, "event": event_cols}
