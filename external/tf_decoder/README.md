# TF-decoder (external)

Trace pool generation (subsection 4.3, `src/step4_trace_pool_generation.py`)
calls into the TF-decoder repository rather than retraining it:

    git clone <TF-decoder repo URL> external/tf_decoder/TF-decoder
    cd external/tf_decoder/TF-decoder
    pip install -r requirements.txt

`step4_trace_pool_generation.py`'s own docstring documents the exact
data layout it expects (`data_{dataset}_rank{N}/<subfolder>/...`) and the
checkpoint/generation CLI it calls. This repo already ships the pools that
script would produce (`data/concept_pools/`), so cloning the TF-decoder
repo is only needed if you want to regenerate them from scratch, e.g. with
a different seed, cluster, or hyperparameters.
