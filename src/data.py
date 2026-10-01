from pathlib import Path
import pandas as pd


def read_csv_with_encoding(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"지원하는 인코딩으로 읽을 수 없습니다: {path}")


def load_raw_data(data_dir: Path):
    normal_path = data_dir / "press_data_normal.csv"
    abnormal_path = data_dir / "outlier_data.csv"

    if not normal_path.exists() or not abnormal_path.exists():
        raise FileNotFoundError(
            "data/에 press_data_normal.csv와 outlier_data.csv를 배치하세요."
        )

    return read_csv_with_encoding(normal_path), read_csv_with_encoding(abnormal_path)


def create_segments(df: pd.DataFrame, gap_threshold: float = 0.11) -> pd.DataFrame:
    out = df.copy()
    out["TimeStamp"] = pd.to_datetime(out["TimeStamp"], errors="raise")
    out["original_index"] = out.index
    out = out.sort_values("TimeStamp", kind="mergesort").reset_index(drop=True)
    out["time_diff"] = out["TimeStamp"].diff().dt.total_seconds()

    new_segment = out["time_diff"].isna() | (out["time_diff"] > gap_threshold)
    out["segment_id"] = new_segment.astype("int64").cumsum()
    return out
