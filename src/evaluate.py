import numpy as np
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

from .config import (
    FEATURE_COLUMNS,
    N_SPLITS,
    SPLIT_RANDOM_STATES,
    THRESHOLD,
    TUNED_CATBOOST_PARAMS,
)


def evaluate_repeated_group_cv(feature_df: pd.DataFrame):
    X = feature_df[FEATURE_COLUMNS]
    y = feature_df["label"].astype(int)
    groups = feature_df["group_id"]

    window_rows = []
    segment_rows = []

    for seed in SPLIT_RANDOM_STATES:
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=seed,
        )
        oof_parts = []

        for fold, (tr, te) in enumerate(
            splitter.split(X, y, groups),
            start=1,
        ):
            if set(groups.iloc[tr]) & set(groups.iloc[te]):
                raise RuntimeError("group leakage detected")

            model = CatBoostClassifier(**TUNED_CATBOOST_PARAMS)
            model.fit(X.iloc[tr], y.iloc[tr])

            prob = model.predict_proba(X.iloc[te])[:, 1]
            pred = (prob >= THRESHOLD).astype(int)

            fold_df = feature_df.iloc[te][
                ["dataset", "label", "segment_id", "group_id", "window_id"]
            ].copy()
            fold_df["actual_label"] = y.iloc[te].to_numpy()
            fold_df["predicted_label"] = pred
            fold_df["abnormal_probability"] = prob
            fold_df["fold"] = fold
            oof_parts.append(fold_df)

        oof = pd.concat(oof_parts, ignore_index=True)
        yt = oof["actual_label"]
        yp = oof["predicted_label"]
        pr = oof["abnormal_probability"]

        window_rows.append({
            "split_random_state": seed,
            "abnormal_precision": precision_score(yt, yp, zero_division=0),
            "abnormal_recall": recall_score(yt, yp, zero_division=0),
            "abnormal_f1": f1_score(yt, yp, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(yt, yp),
            "pr_auc": average_precision_score(yt, pr),
        })

        seg = (
            oof.groupby(["dataset", "group_id", "segment_id"], sort=True)
            .agg(
                actual_label=("actual_label", "first"),
                mean_abnormal_probability=("abnormal_probability", "mean"),
            )
            .reset_index()
        )

        seg["segment_prediction"] = (
            seg["mean_abnormal_probability"] >= THRESHOLD
        ).astype(int)

        ys = seg["actual_label"]
        ysp = seg["segment_prediction"]

        segment_rows.append({
            "split_random_state": seed,
            "abnormal_precision": precision_score(ys, ysp, zero_division=0),
            "abnormal_recall": recall_score(ys, ysp, zero_division=0),
            "abnormal_f1": f1_score(ys, ysp, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(ys, ysp),
            "false_negative_segment_count": int(
                ((ys == 1) & (ysp == 0)).sum()
            ),
            "false_positive_segment_count": int(
                ((ys == 0) & (ysp == 1)).sum()
            ),
        })

    return pd.DataFrame(window_rows), pd.DataFrame(segment_rows)


def summarize(metrics: pd.DataFrame):
    numeric = metrics.select_dtypes(include=[np.number])
    return numeric.agg(["mean", "std", "min", "max"]).T
