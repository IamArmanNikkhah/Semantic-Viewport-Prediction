import json
import argparse
import numpy as np
from os import path
from pathlib import Path
from detection_type import Detection, SEMANTIC_CLASSES

#CONSTANTS
K = len(SEMANTIC_CLASSES)
ROWS = 6
COLUMNS = 12

# Creating a loader for the data that would be given in either JSON or CSV file

def semantic_data_loader(log_file_path: str | Path):
    log_file_path = Path(log_file_path)
    # getting the log file extension
    _, file_extension = path.splitext(log_file_path.name)
    if file_extension == '.json':
        return parse_json_log(log_file_path)
    elif file_extension == '.csv':
        return parse_csv_log(log_file_path)

#TO DO: Implement JSON log parsing logic
def parse_json_log(log_file_path) -> list[Detection]:
    pass
    
#TO DO: Implement CSV log parsing logic
def parse_csv_log(log_file_path) -> list[Detection]:
    pass

# STEP 2: creating a grid per detection second from the detection list (which is per clip)
# the grid is S[t] in the shape of (K, 6, 12) where K is the number of semantic classes
def accumulate_grids(clip_detections: list[Detection], temperature: float = 1.5) -> dict[int,np.ndarray]:
    S = {} # hold all the grids per second of the clip

    # if there is not a grid for second t, create one, else use the existing one
    for detection in clip_detections:
        t = int(detection.timestamp)
        # if grid for second t does not exist, create one
        if t not in S:
            # creating a grid all set to zero
            S[t] = np.zeros((K, ROWS, COLUMNS))
        
        semantic_number = SEMANTIC_CLASSES.index(detection.identified_semantic_class) # get the correct sementic class to know which grid to update
        
        # getting the row and column in the grid from the tile_id
        row = detection.tile_id // COLUMNS
        col = detection.tile_id % COLUMNS

        # temperature scaling for confidence
        scaled_confidence = detection.confidence ** (1 / temperature)
        S[t][semantic_number, row, col] += scaled_confidence

    return S


def run(log_file_path: Path, debugging: bool = False, temperature: float = 1.5):
    clip = semantic_data_loader(log_file_path)
    grids = accumulate_grids(clip, temperature=temperature)
    debugging_statements(f"Generated grids for clip from file: {log_file_path} grids: {grids.keys()}", debug=debugging)

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
    parser.add_argument("--temperature", type=float, default=1.5,
                        help="Temperature for confidence scaling (default: 1.5)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run(**vars(args))