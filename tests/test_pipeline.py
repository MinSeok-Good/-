import unittest

import pandas as pd

from src.data import create_segments
from src.features import create_window_features


def _sample_df():
    return pd.DataFrame(
        {
            "TimeStamp": [
                "2026-01-01 00:00:00.000",
                "2026-01-01 00:00:00.100",
                "2026-01-01 00:00:00.200",
                "2026-01-01 00:00:02.000",
                "2026-01-01 00:00:02.100",
                "2026-01-01 00:00:02.200",
            ],
            "AI0_Vibration": [1, 2, 3, 4, 5, 6],
            "AI1_Vibration": [1, 2, 3, 4, 5, 6],
            "AI2_Current": [1, 2, 3, 4, 5, 6],
        }
    )


class PipelineBoundaryTests(unittest.TestCase):
    def test_segmentation_splits_on_large_gap(self):
        segmented = create_segments(_sample_df(), gap_threshold=0.11)
        self.assertEqual(segmented["segment_id"].nunique(), 2)
        self.assertEqual(segmented.loc[:2, "segment_id"].nunique(), 1)
        self.assertEqual(segmented.loc[3:, "segment_id"].nunique(), 1)

    def test_window_never_crosses_segment(self):
        segmented = create_segments(_sample_df(), gap_threshold=0.11)
        windows = create_window_features(
            segmented,
            dataset="Test",
            label=0,
            window_size=3,
        )
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows["segment_id"].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
