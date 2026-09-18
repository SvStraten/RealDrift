# RealDrift: Multi-Perspective Drift Generation from Real Event Logs

Reproduces "RealDrift: Multi-Perspective Drift Generation from Real Event
Logs" for BPIC2012 end to end and at full scale: trace clustering (Figure
4), drift composition (recurrent, sudden and gradual tiers), and an
implementation of each of the paper's five pipeline components. Sepsis
follows the same layout once its sublogs and concept pools are added
under `data/`.

## Installation

    pip install -r requirements.txt

Python 3.10+ required. `pix-framework` (used by step1's case-level
attribute classification) requires Python <3.12; on 3.12+ step1 falls
back automatically to a reimplementation of the same method and prints a
warning. See "External repositories" below for the two optional external
clones (AT-KDE, TF-decoder).

## What do you want to do?

| You want to... | Run this |
|---|---|
| See the paper's Figure 4 and Figure 5 reproduced, with plots | `jupyter nbconvert --to notebook --execute --inplace reproduce_bpic12_paper.ipynb`, then open it |
| Confirm every pipeline step works, with assertions | `jupyter nbconvert --to notebook --execute --inplace test_pipeline.ipynb`, then open it |
| Generate a drift log (recurrent tier, full scale) | `python generate_drift_log.py --config configs/bpic2012_config.yaml` |
| Generate a sudden or gradual tier instead | add `--tier sudden` or `--tier gradual` |
| Get a fast sanity-check log before a full run | add `--skip-sa --max-cases-per-instance 400` |
| Compose with a resource calendar constraint | add `--calendars <pickle>` (see step2_resource_calendars.py) |
| Understand what each file in `src/` does | see "Repo layout" below, or each file's own module docstring |

Both notebooks take a few minutes to run top to bottom (see their own
runtime notes in their first cell); that is real computation on real
data, not a placeholder.

## generate_drift_log.py arguments

| Argument | Default | Description |
|---|---|---|
| `--config` | required | Path to a dataset config YAML (e.g. `configs/bpic2012_config.yaml`) |
| `--tier` | `recurrent` | `sudden`, `gradual`, or `recurrent`. sudden/gradual chain one instance per concept; recurrent chains two (A/B), all transitions sudden |
| `--calendars` | none | Path to a pickle of `(calendars, pooled_resources)` from `step2_resource_calendars.to_composer_calendars`. Without it, composition runs with no resource-based feasibility constraint |
| `--max-beds` | none | Optional global bed-capacity condition (subsection 4.4) |
| `--skip-sa` | off | Skip the simulated-annealing refinement stage, greedy pass only |
| `--sa-iterations` | `2000` | Simulated-annealing iteration count when SA is not skipped |
| `--gradual-window` | `30` | Blend window `w` (cases) for `gradual` transitions. Larger `w` starts blending earlier and spreads the transition out further. No effect on `sudden` |
| `--seed` | `0` | Random seed |
| `--out-dir` | `outputs` | Output directory |
| `--max-cases-per-instance` | none | Truncates each instance's pool, for a fast sanity check before a full run |

Output: `outputs/{dataset}_{tier}_drift_log.csv` (the composed event log)
and `outputs/{dataset}_{tier}_transitions.csv` (ground-truth drift
labels, one row per transition). If any cases could not be scheduled
within the feasibility search horizon, `outputs/{dataset}_{tier}_excluded.csv`
lists them.

## What is included vs. what you run yourself

This repo ships:
- The real BPIC2012 sublogs (`data/sublogs/`, one CSV per cluster
  C1..C5, produced by trace clustering).
- The pre-generated trace pools (`data/concept_pools/`, two
  independently-seeded instances A and B per cluster, produced by the
  TF-decoder).
- The raw `BPIC12_complete.xes` log (`data/raw/`, 262,200 events across
  24 activities and 68 resources, SCHEDULE/START/COMPLETE lifecycle
  included), so step1 and step2 run against real data.
- The AT-KDE repository (`external/AT-KDE/`), so step5's arrival model
  can use the global/weekday/time-of-day decomposition instead of only
  the flat-KDE fallback. See "External repositories" for the one
  compatibility patch applied to it.

You do not need to retrain anything to compose a drift log,
`generate_drift_log.py` works directly off the sublogs and pools. The
TF-decoder itself is only needed to regenerate the concept pools from
scratch with different settings.

## External repositories

| Repository | Used by | Required? |
|---|---|---|
| [pix-framework](https://github.com/AutomatedProcessImprovement/pix-framework) | `step1_attribute_classification.py`, case-level attribute discovery | Optional (Python <3.12 only; falls back to a built-in reimplementation otherwise) |
| [AT-KDE](https://github.com/konradoezdemir/AT-KDE) | `step5_arrival_time_modeling.py` / `atkde_adapter.py`, arrival modeling | Optional (`use_atkde: true` in the config; flat-KDE is the default) |
| TF-decoder (Perdikogiannis et al., BPM, to appear) | `step4_trace_pool_generation.py`, trace pool generation | Optional (only to regenerate `data/concept_pools/` from scratch; not yet public, add the URL to `external/tf_decoder/README.md` once released) |
| [explainable_concept_drift_pm](https://github.com/niklasadams/explainable_concept_drift_pm) (PELT) | Not included; drift detection baseline evaluated against this repo's output in the paper | Only for reproducing Table 3 |
| [version-clustering-cdd](https://gitlab.cs.univie.ac.at/bernolda00cs/version-clustering-cdd) (VC) | Not included; drift detection baseline | Only for reproducing Table 3 |
| [COMPASS](https://github.com/SvStraten/COMPASS) | Not included; continual learning baseline | Only for reproducing Figure 5's accuracy overlay |

Setup for AT-KDE:

    git clone https://github.com/konradoezdemir/AT-KDE.git external/AT-KDE
    pip install KDEpy

NumPy 2.0 removed `np.infty`; the AT-KDE repo uses it in
`source/iat_approaches/kde.py` and `diagnostics/eval_event_logs.py`.
Replace both occurrences with `np.inf` after cloning if running on
NumPy 2.x (already patched in the copy under `external/AT-KDE/` if it
was included with this repo).

Setup for pix-framework (Python <3.12 only):

    pip install pix-framework

The other three (PELT, VC, COMPASS) are not called by anything in this
repo; they consume this repo's output (`outputs/*_drift_log.csv` and
`outputs/*_transitions.csv`) as their own input if you want to reproduce
the paper's detection and continual-learning results.

## Repository Structure

    configs/bpic2012_config.yaml           dataset paths, t0, use_atkde flag
    data/raw/BPIC12_complete.xes           the real log (step1, step2 run against this)
    data/sublogs/                          real per-concept sublogs (arrival fitting, step3 output)
    data/concept_pools/                    pre-generated trace pools, two instances (A/B) per concept
    external/AT-KDE/                       AT-KDE repo (step5's non-default arrival model)
    external/tf_decoder/README.md          pointer to the TF-decoder repo (step4, not included)

    src/step1_attribute_classification.py  subsection 4.1 (attributes): case-level via a
                                            pix-framework-compatible method (falls back to a
                                            reimplementation on Python 3.12+); global-vs-case and
                                            event splits are this repo's own method, see the
                                            module's own docstring
    src/step2_resource_calendars.py        subsection 4.1 (calendars): confidence/support-based
                                            weekly calendar discovery per resource
    src/step3_trace_clustering.py          subsection 4.2: feature extraction, PCA, K-means,
                                            silhouette-based or fixed-K clustering
    src/step4_trace_pool_generation.py     subsection 4.3: TF-decoder wrapper, calls the external
                                            TF-decoder repo, does not train or reimplement it
    src/step5_arrival_time_modeling.py     subsection 4.4 (arrival half): flat-KDE baseline, or
                                            AT-KDE via atkde_adapter.py
    src/step6_drift_composer.py            subsection 4.4 (Feasible(), global conditions) and 4.5
                                            (the drift composer): greedy pass + simulated
                                            annealing, with a bisect-based occupancy index (see
                                            "Performance" below)
    src/atkde_adapter.py                   wrapper around external/AT-KDE/, not a reimplementation
    src/io_utils.py                        shared CSV loading/export utilities steps 3-6 depend on
    src/legacy_simple_composer.py          simplified composer without SA refinement, kept for
                                            reference, not used by generate_drift_log.py

    generate_drift_log.py                  orchestrator, this is what you run for a log
    test_pipeline.ipynb                    runs and asserts on every step above, on real data
    reproduce_bpic12_paper.ipynb           reproduces Figure 4 and the recurrent/sudden/gradual tiers
    outputs/                               generated logs, transition ground truth, and figures land here
