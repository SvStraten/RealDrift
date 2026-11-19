"""Central configuration for the concept drift detection pipeline:
distance measure and evaluation settings (lag, mode)."""

##### DATA CONFIG #####
DISTANCE_MEASURE = "cos" # can be one of ["fro","nuc","inf","l2","cos","earth"]

# Evaluation config
RELATIVE_LAG = [0.05, 0.01, 0.025]
EVAL_MODE = "general"