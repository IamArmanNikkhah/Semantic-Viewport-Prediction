import argparse
import json
import sys
from os import path
from pathlib import Path
import pandas as pd
import numpy as np
from ingest.log_info import LogInfo, FrameData
from pathlib import Path

# CONSTANTS
RADIANS = np.pi / 180

'''
    running the script: if the log is JSON file parse using json parsing function,
    if log is a csv file parse using csv parsing function

    JSON log file format example from the AVTrack360 dataset:
    {
    "data":[{  // Containing the different captured data of the videos in an array
        "filename":"2.mp4",  // The name of the played back video file.
        "hmd":"vive",  // The name of the used HMD.
        "pitch_yaw_roll_data_hmd":[  // Containing the captured pitch/yaw/roll data and the playback time of the video ("sec").
            {
                "pitch":-13,
                "roll":2,
                "sec":0.266,
                "yaw":0
            },
            ...
        ],
    ...
    ]
}
'''

def parse_log_file(log_file_path: str | Path):
    log_file_path = Path(log_file_path)
    # getting the log file extension
    _, file_extension = path.splitext(log_file_path.name)
    if file_extension == '.json':
        return parse_json_log(log_file_path)
    elif file_extension == '.csv':
        return parse_csv_log(log_file_path)

# This function parses JSON log files


def parse_json_log(log_file_path) -> list[LogInfo]:
    # opening the JSON file
    with open(log_file_path, "r") as f:
        raw_logs = json.load(f)

    parsed_logs = []
    for item in raw_logs.get("data", []):
        frames = [
            # creating FrameData dataclass for each frame in the log
            FrameData(
                pitch=int(f["pitch"]),
                roll=float(f["roll"]),
                sec=float(f["sec"]),
                yaw=float(f["yaw"])
            )
            for f in item.get("pitch_yaw_roll_data_hmd", [])
        ]
        # creating LogInfo dataclass for each log item
        log = LogInfo(
            filename=item.get("filename", ""),
            hmd=item.get("hmd", ""),
            data=frames,
            log_type="json",
            label=raw_logs.get("label"),
            video_length_s=item.get("video_length_in_s")
        )
        parsed_logs.append(log)
    return parsed_logs

# NOT IMPLEMENETED YET


def parse_csv_log(log_file_path) -> list[LogInfo]:
    """TODO: Implement CSV log parsing logic."""""
    raise NotImplementedError("CSV log parsing not yet implemented")


def normalize_logs(logs: list[LogInfo]) -> list[LogInfo]:
    for log in logs:
        for frame in log.data:
            # Sign convention:
            # turning left = negative yaw
            # pitching up = negative pitch.

            # converting to radians (deg * pi/180 = rad)
            pitch_rad = float(frame.pitch) * RADIANS
            yaw_rad = float(frame.yaw) * RADIANS
            roll_rad = float(frame.roll) * RADIANS

            # warp yaw is in range (-pi,pi] by centering at 0
            yaw_wrapped = ((yaw_rad + np.pi) % (2 * np.pi)) - np.pi

            # pitch is clamped between [-pi/2 to pi/2] by centering at 0
            pitch_clamped = max(-np.pi / 2, min(np.pi / 2, pitch_rad))

            # updating the frame data with all the normazlied values
            frame.pitch = pitch_clamped
            frame.yaw = yaw_wrapped
            frame.roll = roll_rad
    return logs


# saving the data of the parsed log into a paraquet file
def save_parsed_logs(parsed_logs: list[LogInfo], file_path_name: str):
    user = Path(file_path_name).stem  # get the filename without extension
    for log in parsed_logs:
        # get the video filename without extension
        clip = Path(log.filename).stem
        output_path = Path(f"data/standardized/user{user}_clip{clip}.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for frame in log.data:
            rows.append({
                "user": user,
                "clip": clip,
                "filename": log.filename,
                "hmd": log.hmd,
                "pitch": frame.pitch,
                "roll": frame.roll,
                "sec": frame.sec,
                "yaw": frame.yaw,
                "log_type": log.log_type,
                "label": log.label,
                "video_length_s": log.video_length_s,
            })

        df = pd.DataFrame(rows)
        df.to_parquet(output_path, index=False)

def resample_to_60hz(parquet_path: Path):
    """Create a uniform 60 Hz timeline and interpolate pitch/roll/yaw."""
    df = pd.read_parquet(parquet_path)

    # Raw time
    t_raw = df["sec"].to_numpy()
    duration = df["video_length_s"].iloc[0]
    t_uniform = np.arange(0, duration, 1/60.0)  # 60 Hz timeline

    # Handle yaw, interpolation
    yaw_raw = df["yaw"].to_numpy()
    yaw_rad = np.deg2rad(yaw_raw) if np.max(np.abs(yaw_raw)) > np.pi else yaw_raw
    yaw_unwrapped = np.unwrap(yaw_rad)
    yaw_interp = np.interp(t_uniform, t_raw, yaw_unwrapped)
    yaw_resampled = ((yaw_interp + np.pi) % (2*np.pi)) - np.pi

    # Interpolate pitch and roll
    pitch_resampled = np.interp(t_uniform, t_raw, df["pitch"])
    roll_resampled  = np.interp(t_uniform, t_raw, df["roll"])

    df_60hz = pd.DataFrame({
        "time_s": t_uniform,
        "pitch": pitch_resampled,
        "roll":  roll_resampled,
        "yaw":   yaw_resampled
    })

    out_path = parquet_path.with_name(parquet_path.stem + "_60hz.parquet")
    df_60hz.to_parquet(out_path, index=False)
    print(f"✅ 60 Hz resampled data saved to {out_path}")


def calculate_angular_velocity(parquet_path):
    """ Calculate angular velocities (radians/sec) for pitch, roll, and yaw on resampled 60 Hz data."""

    df = pd.read_parquet(parquet_path)

    # Constant time step
    dt = 1 / 60.0

    # Angular velocity for each angle
    df["pitch_velocity"] = df["pitch"].diff() / dt
    df["roll_velocity"]  = df["roll"].diff() / dt

    # For yaw
    dyaw = np.diff(df["yaw"].to_numpy(), prepend=df["yaw"].iloc[0])
    dyaw = (dyaw + np.pi) % (2*np.pi) - np.pi  # unwrap tiny jumps
    df["yaw_velocity"] = dyaw / dt

    df.iloc[0, df.columns.get_loc("pitch_velocity")] = 0
    df.iloc[0, df.columns.get_loc("roll_velocity")]  = 0
    df.iloc[0, df.columns.get_loc("yaw_velocity")]   = 0

    out_path = parquet_path.with_name(parquet_path.stem + "_vel.parquet")
    df.to_parquet(out_path, index=False)
    print(f"✅ Angular velocities saved to: {out_path}")

    return df



def run(log_file_path: Path, debugging: bool = False):
    debugging_statements(f"Parsing log file: {log_file_path}", debugging)
    parsed_logs = parse_log_file(log_file_path)

    debugging_statements("Normalizing log angles", debugging)
    normalized_logs = normalize_logs(parsed_logs)

    save_parsed_logs(normalized_logs, log_file_path)
    debugging_statements(f"Saved parsed logs", debugging)

    user = Path(log_file_path).stem
    clip = Path(parsed_logs[0].filename).stem
    parquet_path = Path(f"data/standardized/user{user}_clip{clip}.parquet")
    resample_to_60hz(parquet_path)

    vel_source = parquet_path.with_name(parquet_path.stem + "_60hz.parquet")
    calculate_angular_velocity(vel_source)

# function for debugging statements on/off


def debugging_statements(message: str, debug: bool = False):
    if debug:
        print(f"[DEBUG] {message}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Parse log files and save as parquet.")
    parser.add_argument("log_file_path", type=Path,
                        help="Path to the log file to parse.")
    parser.add_argument("--debugging", action="store_true",
                        help="Enable debugging statements output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run(**vars(args))
