"""
download_and_prepare.py

One-shot entrypoint for Week 1:
- Runs AVTrack360 ingestion for a given raw log.
- Generates the alignment report for a given <user, clip>.
- Optionally prepares the dev subset folder.

Usage (from repo root):

    python scripts/download_and_prepare.py --dev-subset

By default this targets:
- raw log: data/raw/avtrack360/10.json
- user:    10
- clip:    6
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    """Run a subprocess command and raise if it fails."""
    print(f"👉 Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_ingest(raw_log: Path) -> None:
    """Call the AVTrack360 ingestion script on the given raw log."""
    cmd = [
        sys.executable,
        "-m",
        "ingest.scripts.avtrack360_loader",
        str(raw_log),
        "--debugging",
    ]
    run(cmd)


def run_alignment(raw_log: Path, user: str, clip: str) -> None:
    """Call the alignment report script for <user, clip>."""
    cmd = [
        sys.executable,
        "-m",
        "ingest.scripts.make_alignment_report",
        str(raw_log),
        "--user",
        str(user),
        "--clip",
        str(clip),
    ]
    run(cmd)


def prepare_dev_subset(user: str, clip: str) -> None:
    """
    Copy raw + standardized + reports into data/dev/...
    for the selected <user, clip>.
    """
    user = str(user)
    clip = str(clip)

    raw_log = REPO_ROOT / "data" / "raw" / "avtrack360" / f"{user}.json"

    std_dir = REPO_ROOT / "data" / "standardized"
    base_std = std_dir / f"user{user}_clip{clip}.parquet"
    hz60_std = std_dir / f"user{user}_clip{clip}_60hz.parquet"
    vel_std = std_dir / f"user{user}_clip{clip}_60hz_vel.parquet"

    report_dir = REPO_ROOT / "reports" / "alignment" / f"user{user}_clip{clip}"

    dev_root = REPO_ROOT / "data" / "dev" / f"avtrack360_user{user}_clip{clip}"
    dev_raw = dev_root / "raw"
    dev_std = dev_root / "standardized"
    dev_rep = dev_root / "reports"

    print(f"👉 Preparing dev subset under: {dev_root}")
    dev_raw.mkdir(parents=True, exist_ok=True)
    dev_std.mkdir(parents=True, exist_ok=True)
    dev_rep.mkdir(parents=True, exist_ok=True)

    # Copy raw log
    shutil.copy2(raw_log, dev_raw / raw_log.name)

    # Copy standardized parquets
    for src in [base_std, hz60_std, vel_std]:
        if not src.exists():
            raise FileNotFoundError(f"Expected standardized file missing: {src}")
        shutil.copy2(src, dev_std / src.name)

    # Copy alignment report artifacts
    if not report_dir.exists():
        raise FileNotFoundError(f"Expected report directory missing: {report_dir}")

    for src in report_dir.iterdir():
        if src.is_file():
            shutil.copy2(src, dev_rep / src.name)

    print("✅ Dev subset prepared.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-shot ingestion + alignment + optional dev subset for AVTrack360."
    )
    p.add_argument(
        "--raw-log",
        type=str,
        default="data/raw/avtrack360/10.json",
        help="Path to AVTrack360 raw JSON log (default: data/raw/avtrack360/10.json)",
    )
    p.add_argument(
        "--user",
        type=str,
        default="10",
        help="User id (default: 10)",
    )
    p.add_argument(
        "--clip",
        type=str,
        default="6",
        help="Clip id (default: 6)",
    )
    p.add_argument(
        "--dev-subset",
        action="store_true",
        help="If set, also prepares the dev subset under data/dev/.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_log = REPO_ROOT / args.raw_log
    user = args.user
    clip = args.clip

    if not raw_log.exists():
        raise FileNotFoundError(f"Raw log not found: {raw_log}")

    print("=== Step 1: Ingest AVTrack360 raw log ===")
    run_ingest(raw_log)

    print("\n=== Step 2: Generate alignment report ===")
    run_alignment(raw_log, user=user, clip=clip)

    if args.dev_subset:
        print("\n=== Step 3: Prepare dev subset ===")
        prepare_dev_subset(user=user, clip=clip)


if __name__ == "__main__":
    main()
