from pathlib import Path

import numpy as np
import pandas as pd


# This assumes you've already run:
#   python scripts/download_and_prepare.py --dev-subset
DATA_ROOT = Path("data/dev/avtrack360_user10_clip6")


def test_standardized_parquet_schema_and_types():
    """Check base standardized file has the right columns and numeric types."""
    p = DATA_ROOT / "standardized" / "user10_clip6.parquet"
    assert p.exists(), f"Expected standardized parquet not found: {p}"

    df = pd.read_parquet(p)

    # Adjust this list if your actual column names differ.
    required_cols = ["sec", "pitch", "yaw", "roll"]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # All required columns should be numeric
    for col in required_cols:
        assert np.issubdtype(
            df[col].dtype, np.number
        ), f"Column {col} must be numeric, got {df[col].dtype}"

    # Time must be strictly increasing
    sec = df["sec"].to_numpy()
    assert np.all(np.diff(sec) > 0), "sec must be strictly increasing"


def test_angle_ranges_are_sane():
    """Check that angles are within reasonable radian ranges after normalization."""
    p = DATA_ROOT / "standardized" / "user10_clip6.parquet"
    assert p.exists(), f"Expected standardized parquet not found: {p}"

    df = pd.read_parquet(p)

    pitch = df["pitch"].to_numpy()
    yaw = df["yaw"].to_numpy()
    roll = df["roll"].to_numpy()

    # Pitch in [-pi/2, pi/2]
    assert pitch.min() >= -np.pi / 2 - 1e-3
    assert pitch.max() <= np.pi / 2 + 1e-3

    # Yaw in [-pi, pi]
    assert yaw.min() >= -np.pi - 1e-3
    assert yaw.max() <= np.pi + 1e-3

    # Roll typically in [-pi, pi] as well
    assert np.abs(roll).max() <= np.pi + 1e-3
.ta