import argparse
from pathlib import Path

from catboost import CatBoostClassifier

from .config import (
    DEFAULT_DATA_DIR,
    FEATURE_COLUMNS,
    GAP_THRESHOLD_SECONDS,
    TUNED_CATBOOST_PARAMS,
    WINDOW_SIZE,
)
from .data import create_segments, load_raw_data
from .evaluate import evaluate_repeated_group_cv, summarize
from .features import build_feature_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    normal, abnormal = load_raw_data(args.data_dir)
    normal_seg = create_segments(normal, GAP_THRESHOLD_SECONDS)
    abnormal_seg = create_segments(abnormal, GAP_THRESHOLD_SECONDS)

    feature_df = build_feature_dataset(
        normal_seg,
        abnormal_seg,
        WINDOW_SIZE,
    )

    window_metrics, segment_metrics = evaluate_repeated_group_cv(feature_df)

    print("\nWindow-level repeated Group CV")
    print(summarize(window_metrics))

    print("\nSegment-level mean-probability aggregation")
    print(summarize(segment_metrics))

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    window_metrics.to_csv(
        results_dir / "window_cv_metrics.csv",
        index=False,
    )
    segment_metrics.to_csv(
        results_dir / "segment_cv_metrics.csv",
        index=False,
    )

    if args.save_model:
        model = CatBoostClassifier(**TUNED_CATBOOST_PARAMS)
        model.fit(
            feature_df[FEATURE_COLUMNS],
            feature_df["label"].astype(int),
        )
        Path("models").mkdir(exist_ok=True)
        model.save_model("models/catboost_press_anomaly.cbm")
        print("Saved models/catboost_press_anomaly.cbm")


if __name__ == "__main__":
    main()
