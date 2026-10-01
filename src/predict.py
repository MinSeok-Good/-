import argparse
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

from .config import FEATURE_COLUMNS, GAP_THRESHOLD_SECONDS, WINDOW_SIZE
from .data import create_segments, read_csv_with_encoding
from .features import create_window_features


def predict_file(input_path: Path, model_path: Path) -> pd.DataFrame:
    raw = read_csv_with_encoding(input_path)

    required = {"TimeStamp", "AI0_Vibration", "AI1_Vibration", "AI2_Current"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"입력 CSV에 필요한 컬럼이 없습니다: {missing}")

    segmented = create_segments(raw, GAP_THRESHOLD_SECONDS)
    features = create_window_features(
        segmented,
        dataset="Inference",
        label=0,
        window_size=WINDOW_SIZE,
    )

    if features.empty:
        raise ValueError(
            f"완전한 {WINDOW_SIZE}-sample window를 만들 수 없습니다."
        )

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    probability = model.predict_proba(features[FEATURE_COLUMNS])[:, 1]
    output = features[
        [
            "segment_id",
            "window_id",
            "window_in_segment",
            "window_start_time",
            "window_end_time",
        ]
    ].copy()
    output["abnormal_probability"] = probability
    output["prediction"] = (probability >= 0.5).astype(int)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/catboost_press_anomaly.cbm"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/predictions.csv"),
    )
    args = parser.parse_args()

    result = predict_file(args.input, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.head().to_string(index=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
