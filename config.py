import os

# The folders are allocated in such a way as to take into account that the repository can be launched on different operating systems
# (the type of directory paths is different)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Based on EDA, the following experimental parameters were identified:
HORIZON = 14 # Forecasting horizon
WINDOW_SIZE = 30 # Historical window size for CatBoost
SEASONALITY = 7

# Parameters of EDA-based models and hypotheses:
N_CLUSTERS = 3
N_SPLITS = 3
TRANSFORMS = ['none', 'log1p', 'boxcox', 'diff']

# Separate hyperparameters for CatBoost
CB_ITERATIONS = 1000 # Maximum number of trees
CB_EARLY_STOPPING = 50
CB_VAL_RATIO = 0.1 # The proportion of recent windows for the validation window