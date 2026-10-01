from pathlib import Path

GAP_THRESHOLD_SECONDS = 0.11
WINDOW_SIZE = 10
SENSOR_COLUMNS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
FEATURE_STATS = ["mean", "std", "rms", "min", "max"]
FEATURE_COLUMNS = [f"{sensor}_{stat}" for sensor in SENSOR_COLUMNS for stat in FEATURE_STATS]

SPLIT_RANDOM_STATES = [0, 1, 2, 3, 4, 10, 20, 42, 100, 2026]
N_SPLITS = 5
THRESHOLD = 0.5

TUNED_CATBOOST_PARAMS = {
    "depth": 4,
    "learning_rate": 0.03,
    "iterations": 400,
    "auto_class_weights": "Balanced",
    "loss_function": "Logloss",
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": -1,
}

DEFAULT_DATA_DIR = Path("data")
