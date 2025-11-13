from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path("data/dev/avtrack360_user10_clip6")


def test_60hz_cadence_is_steady():
    """
    Check that the 60 Hz standardized file has a steady sample clock.
    """
    p = DATA_ROOT / "standardized" / "user10_clip6_60hz.parquet"
    assert p.exists(), f"Expected 60 Hz parquet not found: {p}"

    df = pd.read_parquet(p)
    assert "time_s" in df.columns, "time_s column is missing in 60 Hz parquet"

    t = df["time_s"].to_numpy()
    dt = np.diff(t)

    nominal_dt = 1.0 / 60.0

    # All intervals should be within a small tolerance of nominal 1/60 s.
    # (This is stricter than the '16 or 17 ms' wording — which is fine.)
    assert np.all(
        np.abs(dt - nominal_dt) < 1e-4
    ), f"60 Hz cadence violated: min(dt)={dt.min()}, max(dt)={dt.max()}"
