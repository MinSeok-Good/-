from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from .config import FEATURE_COLUMNS, TUNED_CATBOOST_PARAMS
from .data import create_segments, load_raw_data
from .features import build_feature_dataset


WINDOW_SIZES = [5, 10, 20]
RANDOM_STATE = 42
N_SPLITS = 5


def evaluate_window_size(normal_seg, abnormal_seg, window_size):
    feature_df = build_feature_dataset(
        normal_seg,
        abnormal_seg,
        window_size=window_size,
    )

    X = feature_df[FEATURE_COLUMNS]
    y = feature_df["label"].astype(int)
    groups = feature_df["group_id"]

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups),
        start=1,
    ):
        if set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]):
            raise RuntimeError("group leakage detected")

        model = CatBoostClassifier(**TUNED_CATBOOST_PARAMS)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        yt = y.iloc[test_idx]

        rows.append(
            {
                "window_size": window_size,
                "fold": fold,
                "precision": precision_score(yt, pred, zero_division=0),
                "recall": recall_score(yt, pred, zero_division=0),
                "f1": f1_score(yt, pred, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(yt, pred),
                "pr_auc": average_precision_score(yt, prob),
            }
        )

    fold_df = pd.DataFrame(rows)
    summary = {
        "window_size": window_size,
        "window_seconds_approx": window_size / 10,
        "total_windows": len(feature_df),
        "normal_windows": int((feature_df["label"] == 0).sum()),
        "abnormal_windows": int((feature_df["label"] == 1).sum()),
        "normal_usable_segments": feature_df.loc[
            feature_df["label"] == 0, "group_id"
        ].nunique(),
        "abnormal_usable_segments": feature_df.loc[
            feature_df["label"] == 1, "group_id"
        ].nunique(),
        "precision_mean": fold_df["precision"].mean(),
        "recall_mean": fold_df["recall"].mean(),
        "f1_mean": fold_df["f1"].mean(),
        "f1_std": fold_df["f1"].std(),
        "balanced_accuracy_mean": fold_df["balanced_accuracy"].mean(),
        "pr_auc_mean": fold_df["pr_auc"].mean(),
        "pr_auc_std": fold_df["pr_auc"].std(),
    }
    return fold_df, summary


def main():
    normal, abnormal = load_raw_data(Path("data"))
    normal_seg = create_segments(normal)
    abnormal_seg = create_segments(abnormal)

    fold_parts = []
    summaries = []

    for window_size in WINDOW_SIZES:
        fold_df, summary = evaluate_window_size(
            normal_seg,
            abnormal_seg,
            window_size,
        )
        fold_parts.append(fold_df)
        summaries.append(summary)

    fold_results = pd.concat(fold_parts, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    Path("results").mkdir(exist_ok=True)
    fold_results.to_csv(
        "results/window_sensitivity_fold_metrics.csv",
        index=False,
    )
    summary_df.to_csv(
        "results/window_sensitivity_summary.csv",
        index=False,
    )

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
