import numpy as np
import pandas as pd

from .config import SENSOR_COLUMNS


def _rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def create_window_features(
    segmented_df: pd.DataFrame,
    dataset: str,
    label: int,
    window_size: int = 10,
):
    rows = []
    window_id = 1

    for segment_id, segment in segmented_df.groupby("segment_id", sort=True):
        segment = (
            segment.sort_values("TimeStamp", kind="mergesort")
            .reset_index(drop=True)
        )
        n_windows = len(segment) // window_size

        for w in range(n_windows):
            chunk = segment.iloc[w * window_size:(w + 1) * window_size]
            row = {
                "dataset": dataset,
                "label": int(label),
                "segment_id": int(segment_id),
                "window_id": window_id,
                "window_in_segment": w + 1,
                "window_start_time": chunk["TimeStamp"].iloc[0],
                "window_end_time": chunk["TimeStamp"].iloc[-1],
            }

            for sensor in SENSOR_COLUMNS:
                x = chunk[sensor].astype(float)
                row[f"{sensor}_mean"] = float(x.mean())
                row[f"{sensor}_std"] = float(x.std(ddof=1))
                row[f"{sensor}_rms"] = _rms(x)
                row[f"{sensor}_min"] = float(x.min())
                row[f"{sensor}_max"] = float(x.max())

            rows.append(row)
            window_id += 1

    return pd.DataFrame(rows)


def build_feature_dataset(
    normal_segmented,
    abnormal_segmented,
    window_size: int = 10,
):
    normal = create_window_features(
        normal_segmented, "Normal", 0, window_size
    )
    abnormal = create_window_features(
        abnormal_segmented, "Abnormal", 1, window_size
    )

    data = pd.concat([normal, abnormal], ignore_index=True)
    data["group_id"] = (
        data["dataset"] + "_segment_" + data["segment_id"].astype(str)
    )
    return data
