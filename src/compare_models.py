from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS
from .data import create_segments, load_raw_data
from .features import build_feature_dataset


RANDOM_STATE = 42
N_SPLITS = 5
THRESHOLD = 0.5


def create_models():
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            loss_function="Logloss",
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        ),
    }


def evaluate_models(feature_df: pd.DataFrame) -> pd.DataFrame:
    X = feature_df[FEATURE_COLUMNS]
    y = feature_df["label"].astype(int)
    groups = feature_df["group_id"]

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    splits = list(splitter.split(X, y, groups))

    rows = []
    for model_name, model in create_models().items():
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            train_groups = set(groups.iloc[train_idx])
            test_groups = set(groups.iloc[test_idx])
            overlap = train_groups & test_groups
            if overlap:
                raise RuntimeError(
                    f"Fold {fold}: group leakage detected: {len(overlap)} groups"
                )

            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            prob = model.predict_proba(X.iloc[test_idx])[:, 1]
            pred = (prob >= THRESHOLD).astype(int)
            yt = y.iloc[test_idx]

            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "precision": precision_score(yt, pred, zero_division=0),
                    "recall": recall_score(yt, pred, zero_division=0),
                    "f1": f1_score(yt, pred, zero_division=0),
                    "balanced_accuracy": balanced_accuracy_score(yt, pred),
                    "roc_auc": roc_auc_score(yt, prob),
                    "pr_auc": average_precision_score(yt, prob),
                    "false_negative_count": int(((yt == 1) & (pred == 0)).sum()),
                    "false_positive_count": int(((yt == 0) & (pred == 1)).sum()),
                    "group_overlap_count": 0,
                }
            )

    return pd.DataFrame(rows)


def main():
    normal, abnormal = load_raw_data(Path("data"))
    normal_seg = create_segments(normal)
    abnormal_seg = create_segments(abnormal)
    feature_df = build_feature_dataset(normal_seg, abnormal_seg, window_size=10)

    fold_metrics = evaluate_models(feature_df)
    summary = (
        fold_metrics.groupby("model")
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            roc_auc=("roc_auc", "mean"),
            pr_auc=("pr_auc", "mean"),
            false_negative_count=("false_negative_count", "mean"),
            false_positive_count=("false_positive_count", "mean"),
        )
        .reset_index()
        .sort_values("f1", ascending=False)
    )

    Path("results").mkdir(exist_ok=True)
    fold_metrics.to_csv("results/model_comparison_fold_metrics.csv", index=False)
    summary.to_csv("results/model_comparison_summary.csv", index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
