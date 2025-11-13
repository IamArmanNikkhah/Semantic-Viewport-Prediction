"""
Alignment report for AVTrack360 logs.

Given a raw AVTrack360 JSON log and a (user, clip) pair,
this script:

1. Loads the raw log (deg, sec) for the selected clip.
2. Loads standardized parquet & 60 Hz resampled parquet.
3. Computes:
   - Sampling interval histogram (raw log)
   - Drift plot (raw timestamp vs sample index)
   - Angle sanity ranges (deg + normalized rad)
   - Cadence check for 60 Hz file (time_s deltas)
   - Missing data stats (based on large time gaps)
4. Saves:
   - alignment summary JSON
   - a PNG with the main plots
   - a small summary.md

Usage (from repo root):

  python -m ingest.scripts.make_alignment_report ^
      data/raw/avtrack360/10.json --clip 6 --user 10
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ingest.scripts import avtrack360_loader


@dataclass
class AngleStats:
    min: float
    max: float
    mean: float
    std: float


@dataclass
class CadenceStats:
    dt_mean: float
    dt_std: float
    dt_min: float
    dt_max: float
    n_intervals: int
    n_out_of_tolerance: int
    tolerance: float


@dataclass
class AlignmentSummary:
    user: str
    clip: str

    # raw log
    n_raw_samples: int
    raw_start_s: float
    raw_end_s: float
    video_length_s: float

    raw_dt_stats_ms: CadenceStats
    raw_pitch_deg: AngleStats
    raw_yaw_deg: AngleStats
    raw_roll_deg: AngleStats

    # normalized parquet (radians)
    norm_pitch_rad: AngleStats
    norm_yaw_rad: AngleStats
    norm_roll_rad: AngleStats

    # 60 Hz resampled cadence
    hz60_dt_stats_s: CadenceStats


def _angle_stats(arr: np.ndarray) -> AngleStats:
    return AngleStats(
        min=float(arr.min()),
        max=float(arr.max()),
        mean=float(arr.mean()),
        std=float(arr.std()),
    )


def _cadence_stats(dt: np.ndarray, tolerance: float) -> CadenceStats:
    if len(dt) == 0:
        return CadenceStats(
            dt_mean=float("nan"),
            dt_std=float("nan"),
            dt_min=float("nan"),
            dt_max=float("nan"),
            n_intervals=0,
            n_out_of_tolerance=0,
            tolerance=float(tolerance),
        )

    return CadenceStats(
        dt_mean=float(dt.mean()),
        dt_std=float(dt.std()),
        dt_min=float(dt.min()),
        dt_max=float(dt.max()),
        n_intervals=int(len(dt)),
        n_out_of_tolerance=int(np.sum(np.abs(dt - dt.mean()) > tolerance)),
        tolerance=float(tolerance),
    )


def _select_log_for_clip(logs, clip: str):
    """Find the LogInfo entry whose filename matches '<clip>.mp4'."""
    target = f"{clip}.mp4"
    for log in logs:
        if Path(log.filename).name == target:
            return log
    raise ValueError(f"Clip {clip} not found in log file (looked for {target}).")


def build_alignment_report(
    log_file_path: Path, user: str, clip: str, output_dir: Path
) -> AlignmentSummary:
    # 1) Load raw JSON log (degrees, sec)
    logs = avtrack360_loader.parse_json_log(log_file_path)
    log = _select_log_for_clip(logs, clip)

    raw_t = np.array([frame.sec for frame in log.data], dtype=float)
    raw_pitch_deg = np.array([frame.pitch for frame in log.data], dtype=float)
    raw_yaw_deg = np.array([frame.yaw for frame in log.data], dtype=float)
    raw_roll_deg = np.array([frame.roll for frame in log.data], dtype=float)

    if raw_t.size == 0:
        raise RuntimeError("No samples found for selected clip in raw log.")

    # 2) Load standardized parquet (normalized radians)
    std_path = Path(f"data/standardized/user{user}_clip{clip}.parquet")
    if not std_path.exists():
        raise FileNotFoundError(f"Expected standardized parquet at {std_path}")

    df_std = pd.read_parquet(std_path)

    norm_pitch_rad = df_std["pitch"].to_numpy(dtype=float)
    norm_yaw_rad = df_std["yaw"].to_numpy(dtype=float)
    norm_roll_rad = df_std["roll"].to_numpy(dtype=float)

    # 3) Load 60 Hz resampled parquet
    hz60_path = std_path.with_name(std_path.stem + "_60hz.parquet")
    if not hz60_path.exists():
        raise FileNotFoundError(f"Expected 60 Hz parquet at {hz60_path}")

    df_60 = pd.read_parquet(hz60_path)
    t60 = df_60["time_s"].to_numpy(dtype=float)

    # -------------------------------------------------------
    # Sampling histogram (raw)
    # -------------------------------------------------------
    raw_dt = np.diff(raw_t)
    raw_dt_ms = raw_dt * 1000.0

    # -------------------------------------------------------
    # Missing data stats (based on big gaps)
    # -------------------------------------------------------
    if raw_dt_ms.size > 0:
        median_dt_ms = float(np.median(raw_dt_ms))
        gap_factor = 2.0
        gaps = raw_dt_ms > (gap_factor * median_dt_ms)
        n_gaps = int(gaps.sum())
        missing_fraction = n_gaps / max(1, len(raw_dt_ms))
    else:
        median_dt_ms = float("nan")
        gap_factor = 2.0
        n_gaps = 0
        missing_fraction = 0.0

    missing_data_stats = {
        "median_dt_ms": median_dt_ms,
        "gap_factor": gap_factor,
        "n_intervals": int(len(raw_dt_ms)),
        "n_gaps": n_gaps,
        "missing_fraction": float(missing_fraction),
    }

    # -------------------------------------------------------
    # 60 Hz cadence check
    # -------------------------------------------------------
    dt60 = np.diff(t60)
    nominal_dt = 1.0 / 60.0
    hz_tolerance = 1e-4  # seconds

    # -------------------------------------------------------
    # Angle sanity stats
    # -------------------------------------------------------
    raw_pitch_stats = _angle_stats(raw_pitch_deg)
    raw_yaw_stats = _angle_stats(raw_yaw_deg)
    raw_roll_stats = _angle_stats(raw_roll_deg)

    norm_pitch_stats = _angle_stats(norm_pitch_rad)
    norm_yaw_stats = _angle_stats(norm_yaw_rad)
    norm_roll_stats = _angle_stats(norm_roll_rad)

    raw_dt_stats = _cadence_stats(raw_dt_ms, tolerance=5.0)  # ms tolerance
    hz60_dt_stats = _cadence_stats(dt60, tolerance=hz_tolerance)

    # -------------------------------------------------------
    # Build summary dataclass
    # -------------------------------------------------------
    summary = AlignmentSummary(
        user=str(user),
        clip=str(clip),
        n_raw_samples=int(len(raw_t)),
        raw_start_s=float(raw_t[0]),
        raw_end_s=float(raw_t[-1]),
        video_length_s=float(
            log.video_length_s or df_std["video_length_s"].iloc[0]
        ),
        raw_dt_stats_ms=raw_dt_stats,
        raw_pitch_deg=raw_pitch_stats,
        raw_yaw_deg=raw_yaw_stats,
        raw_roll_deg=raw_roll_stats,
        norm_pitch_rad=norm_pitch_stats,
        norm_yaw_rad=norm_yaw_stats,
        norm_roll_rad=norm_roll_stats,
        hz60_dt_stats_s=hz60_dt_stats,
    )

    # -------------------------------------------------------
    # Plotting
    # -------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (1) Histogram of raw sampling interval (ms)
    ax = axes[0, 0]
    ax.hist(raw_dt_ms, bins=40)
    ax.set_title("Raw sampling interval histogram")
    ax.set_xlabel("Δt (ms)")
    ax.set_ylabel("Count")

    # (2) Drift plot: raw timestamp vs sample index
    ax = axes[0, 1]
    sample_idx = np.arange(len(raw_t))
    ax.plot(sample_idx, raw_t, marker=".", linestyle="none", alpha=0.6)
    ax.set_title("Raw timestamp vs sample index")
    ax.set_xlabel("sample index")
    ax.set_ylabel("time (s)")

    # (3) Angle sanity: raw deg vs time
    ax = axes[1, 0]
    ax.plot(raw_t, raw_yaw_deg, label="yaw (deg)", alpha=0.8)
    ax.plot(raw_t, raw_pitch_deg, label="pitch (deg)", alpha=0.8)
    ax.plot(raw_t, raw_roll_deg, label="roll (deg)", alpha=0.8)
    ax.legend(loc="best")
    ax.set_title("Raw angles vs time (deg)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("deg")

    # (4) 60 Hz cadence: dt over time
    ax = axes[1, 1]
    t_center = (t60[1:] + t60[:-1]) / 2.0
    ax.plot(t_center, dt60, marker=".", linestyle="none", alpha=0.6)
    ax.axhline(nominal_dt, color="gray", linewidth=1, label="nominal 1/60 s")
    ax.set_title("60 Hz cadence check")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Δt (s)")
    ax.legend(loc="best")

    fig.suptitle(f"Alignment report: user{user}_clip{clip}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_path = output_dir / f"user{user}_clip{clip}_alignment.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    # -------------------------------------------------------
    # Save summary as JSON (+ missing data stats)
    # -------------------------------------------------------
    summary_dict = asdict(summary)
    summary_dict["missing_data_stats"] = missing_data_stats

    summary_path = output_dir / f"user{user}_clip{clip}_alignment.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    # -------------------------------------------------------
    # Save small summary.md
    # -------------------------------------------------------
    summary_md_path = output_dir / f"user{user}_clip{clip}_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Alignment Summary – user{user}_clip{clip}\n\n")
        f.write(
            f"- Raw samples: {summary.n_raw_samples}\n"
        )
        f.write(
            f"- Raw time span: {summary.raw_start_s:.3f}s → "
            f"{summary.raw_end_s:.3f}s "
            f"(video_length_s={summary.video_length_s:.3f})\n\n"
        )

        f.write("## Angle sanity (raw, degrees)\n")
        f.write(
            f"- yaw: {summary.raw_yaw_deg.min:.1f} "
            f"to {summary.raw_yaw_deg.max:.1f}\n"
        )
        f.write(
            f"- pitch: {summary.raw_pitch_deg.min:.1f} "
            f"to {summary.raw_pitch_deg.max:.1f}\n"
        )
        f.write(
            f"- roll: {summary.raw_roll_deg.min:.1f} "
            f"to {summary.raw_roll_deg.max:.1f}\n\n"
        )

        f.write("## Angle sanity (normalized, radians)\n")
        f.write(
            f"- yaw: {summary.norm_yaw_rad.min:.2f} "
            f"to {summary.norm_yaw_rad.max:.2f}\n"
        )
        f.write(
            f"- pitch: {summary.norm_pitch_rad.min:.2f} "
            f"to {summary.norm_pitch_rad.max:.2f}\n"
        )
        f.write(
            f"- roll: {summary.norm_roll_rad.min:.2f} "
            f"to {summary.norm_roll_rad.max:.2f}\n\n"
        )

        f.write("## 60 Hz cadence\n")
        f.write(
            f"- mean Δt: {summary.hz60_dt_stats_s.dt_mean:.6f} s\n"
        )
        f.write(
            f"- min / max Δt: {summary.hz60_dt_stats_s.dt_min:.6f} / "
            f"{summary.hz60_dt_stats_s.dt_max:.6f} s\n"
        )
        f.write(
            f"- out-of-tolerance intervals: "
            f"{summary.hz60_dt_stats_s.n_out_of_tolerance}\n\n"
        )

        f.write("## Missing data (heuristic)\n")
        f.write(
            f"- median raw Δt: {missing_data_stats['median_dt_ms']:.2f} ms\n"
        )
        f.write(
            f"- gaps > {missing_data_stats['gap_factor']}× median: "
            f"{missing_data_stats['n_gaps']} / "
            f"{missing_data_stats['n_intervals']} intervals\n"
        )
        f.write(
            f"- missing_fraction: "
            f"{missing_data_stats['missing_fraction']:.4f}\n"
        )

    # Also print a tiny human-readable snippet
    print(f"✅ Alignment report saved to: {summary_path}")
    print(f"✅ Alignment plots saved to:  {plot_path}")
    print(f"✅ Alignment summary saved to: {summary_md_path}")

    print(
        f"\nRaw sampling: {summary.n_raw_samples} samples, "
        f"{summary.raw_start_s:.3f}s → {summary.raw_end_s:.3f}s "
        f"(video_length_s={summary.video_length_s:.3f})"
    )
    print(
        f"Raw Δt (ms): mean={summary.raw_dt_stats_ms.dt_mean:.2f}, "
        f"min={summary.raw_dt_stats_ms.dt_min:.2f}, "
        f"max={summary.raw_dt_stats_ms.dt_max:.2f}"
    )
    print(
        f"60Hz Δt (s): mean={summary.hz60_dt_stats_s.dt_mean:.6f}, "
        f"min={summary.hz60_dt_stats_s.dt_min:.6f}, "
        f"max={summary.hz60_dt_stats_s.dt_max:.6f}, "
        f"out_of_tol={summary.hz60_dt_stats_s.n_out_of_tolerance}"
    )

    return summary


def parse_args():
    p = argparse.ArgumentParser(
        description="Build alignment report for AVTrack360 head-motion logs."
    )
    p.add_argument(
        "log_file_path",
        type=Path,
        help="Path to raw AVTrack360 JSON log "
             "(e.g., data/raw/avtrack360/10.json)",
    )
    p.add_argument(
        "--user",
        type=str,
        help="User id (e.g., '10'). Defaults to stem of log file.",
    )
    p.add_argument(
        "--clip",
        type=str,
        required=True,
        help="Clip id as in filename (e.g., '6' for 6.mp4).",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("reports/alignment"),
        help="Base directory for report outputs.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    user = args.user or Path(args.log_file_path).stem
    clip = args.clip

    output_dir = args.outdir / f"user{user}_clip{clip}"
    build_alignment_report(
        log_file_path=args.log_file_path,
        user=user,
        clip=clip,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
