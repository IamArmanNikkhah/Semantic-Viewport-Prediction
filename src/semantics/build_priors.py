import json
import argparse
import numpy as np
from os import path
from pathlib import Path
from semantics.detection_type import Detection
from common.interfaces import DEFAULT_CLASSES, TILE_ROWS, TILE_COLS
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import cv2
import torch

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
model.eval()

# CONSTANTS
K = len(DEFAULT_CLASSES)
FRAME_RATE_HZ = 1.0  # desired sampling rate in Hz

# STEP 1: Extract frames from video each in a frame rate
def frame_extractor(video_path: str, output_folder: str, debug: bool = False):
    video = cv2.VideoCapture(video_path)  # opening the video file

    # checking if the video was opened successfully
    if video is None or not video.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    # getting the frames per seconf of the video to know how many frames to skip
    video_fps = video.get(cv2.CAP_PROP_FPS)

    # printing the fps for debugging
    debugging_statements(f"Video {video_path} FPS: {video_fps}", debug=debug)

    # calculating how many frames to skip to extract 1 frame every second
    frames_per_sample = int(video_fps // FRAME_RATE_HZ)

    frame_num = 0
    while True:
        valid_frame, frame = video.read()  # reading a frame from the video

        # breaking the loop if there are no more frames
        if not valid_frame:
            break

        # saving only 1 frame every second using the calculated frames_per_sample
        if frame_num % frames_per_sample == 0:
            cv2.imwrite(f"{output_folder}/frame_{frame_num}.jpg", frame)

        frame_num += 1

    video.release()  # releasing the video file

# every frame should be processed to extract the semantics
def inference_frame(frame_path: str, debug: bool = False):
    frame = Image.open(frame_path)  # opening the image frame
    
    # preparing the inputs for the model
    inputs = processor(text=[list(DEFAULT_CLASSES)],images=frame, return_tensors="pt")

    outputs = model(**inputs)  # getting the model outputs

    target_sizes = torch.tensor([(frame.height, frame.width)])

    results = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=0.1, text_labels=[list(DEFAULT_CLASSES)]
    )
    return results

# Creating a loader for the data that would be given in either JSON or CSV file

def semantic_data_loader(log_file_path: str | Path):
    log_file_path = Path(log_file_path)
    # getting the log file extension
    _, file_extension = path.splitext(log_file_path.name)
    if file_extension == '.json':
        return parse_json_log(log_file_path)
    elif file_extension == '.csv':
        return parse_csv_log(log_file_path)

# TO DO: Implement JSON log parsing logic
def parse_json_log(log_file_path) -> list[Detection]:
    pass
# TO DO: Implement CSV log parsing logic
def parse_csv_log(log_file_path) -> list[Detection]:
    pass

# STEP 2: creating a grid per detection second from the detection list (which is per clip)
# the grid is S[t] in the shape of (K, 6, 12) where K is the number of semantic classes


def accumulate_grids(clip_detections: list[Detection], temperature: float = 1.5, rate_hz: float = 1.0,) -> dict[int, np.ndarray]:
    S = {}  # hold all the grids per second of the clip

    # if there is not a grid for second t, create one, else use the existing one
    for detection in clip_detections:
        t = int(detection.timestamp)
        # if grid for second t does not exist, create one
        if t not in S:
            # creating a grid all set to zero
            S[t] = np.zeros((K, TILE_ROWS, TILE_COLS))

        # get the correct sementic class to know which grid to update
        semantic_number = DEFAULT_CLASSES.index(
            detection.identified_semantic_class)

        # getting the row and column in the grid from the tile_id
        row = detection.tile_id // TILE_COLS
        col = detection.tile_id % TILE_COLS

        # STEP 3: Temperature scaling for confidence
        scaled_confidence = detection.confidence ** (1 / temperature)
        S[t][semantic_number, row, col] += scaled_confidence

    return S

# STEP 4: Temporal smoothing using Exponential Moving Average (EMA)
def smooth_grids(S: dict[int, np.ndarray], alpha: float = 0.6) -> dict[int, np.ndarray]:
    P_sem = {}
    prev = None

    for t in sorted(S.keys()):
        if prev is None:
            # First second has no history, smoothed = raw
            P_sem[t] = S[t]
        else:
            # Blend raw grid with previously smoothed grid
            P_sem[t] = alpha * S[t] + (1 - alpha) * prev

        prev = P_sem[t]

    # returns a dict
    return P_sem


def run(log_file_path: Path, output_folder: Path, debugging: bool = False, temperature: float = 1.5, alpha: float = 0.6, rate_hz: float = 1.0,):
    # extracting frames from raw MP4 file
    debugging_statements("starting frame extraction", debug=debugging)
    frame_extractor(
        video_path=str(log_file_path),
        output_folder=str(output_folder),
        debug=debugging,
    )
    debugging_statements(
        f"Finished extracting frames from {log_file_path} into {output_folder}",
        debug=debugging,
    )
    # clip = semantic_data_loader(log_file_path)
    # S = accumulate_grids(clip, temperature=temperature)
    # P_sem = smooth_grids(S, alpha=alpha)

    # Debug statements
    debugging_statements(
        f"Generated raw S[t] grids for seconds: {list(S.keys())}",
        debug=debugging
    )

    debugging_statements(
        f"Generated smoothed P_sem[t] grids for seconds: {list(P_sem.keys())}",
        debug=debugging
    )

    debugging_statements(
        f"Cadence (rate_hz): {rate_hz}",
        debug=debugging,
    )
def debugging_statements(message: str, debug: bool = False):
    if debug:
        print(f"[DEBUG] {message}")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Parse log files and save as parquet.")
    parser.add_argument("log_file_path", type=Path,
                        help="Path to the log file to parse.")
    parser.add_argument("--output-folder", type=Path, default=Path(
        "extracted_frames"), help="Folder to save extracted frames.")
    parser.add_argument("--debugging", action="store_true",
                        help="Enable debugging statements output")
    parser.add_argument("--temperature", type=float, default=1.5,
                        help="Temperature for confidence scaling (default: 1.5)")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="EMA smoothing factor (default: 0.6)")
    parser.add_argument("--rate_hz", type=float, default=1.0,
                        help="Cadence of semantic priors in Hz (e.g., 0.5, 1.0, 2.0)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run(**vars(args))