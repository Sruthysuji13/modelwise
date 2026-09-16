"""Central configuration and constants for ModelWise."""

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Safety thresholds
MIN_ROWS_FOR_TRAINING = 30
MIN_ROWS_FOR_STABLE_CV = 100
HIGH_CARDINALITY_THRESHOLD = 0.9      # unique/count ratio
CONSTANT_COLUMN_TOLERANCE = 1         # nunique <= this => constant
IMBALANCE_RATIO_THRESHOLD = 0.80      # majority class share that flags imbalance
LEAKAGE_CORR_THRESHOLD = 0.95         # abs correlation with target

# Hyperparameter tuning
TUNING_N_TRIALS = 25          # Optuna trials per model (kept small/practical)
TUNING_N_ITER_RANDOMSEARCH = 20

# Which tuning backend to prefer if both are installed
PREFERRED_TUNER = "optuna"    # "optuna" or "randomsearch"