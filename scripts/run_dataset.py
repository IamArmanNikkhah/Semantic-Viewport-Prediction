import argparse
import glob
import os
import re

import pandas as pd
import torch
from src.modeling.dataset import HeadMotionDataset

"""
Script to run and verify the HeadMotionDataset for a specific video clip.
This script loads standardized motion data and semantic priors
"""
def main():
    parser = argparse.ArgumentParser(description="Process head motion dataset for a specific video clip")
    parser.add_argument("video_number", type=int, help="Video clip number to process")
    parser.add_argument("--seq_len", type=int, default=30, help="Sequence length for motion history")
    args = parser.parse_args()

    video_number = args.video_number
    seq_len = args.seq_len

    # Define paths
    standardized_dir = "data/standardized"
    semantic_path = f"data/semantic_priors/{video_number}.pt"

    # Check if semantic prior exists
    if not os.path.exists(semantic_path):
        print(f"Error: Semantic prior not found at {semantic_path}")
        return

    # Find all parquet files matching the video number pattern
    # Pattern: userX_clipY.parquet or userX_clipY_60hz.parquet
    pattern = os.path.join(standardized_dir, f"user*_clip{video_number}.parquet")
    matching_files = glob.glob(pattern)
    
    # Also check for 60hz files
    pattern_60hz = os.path.join(standardized_dir, f"user*_clip{video_number}_60hz.parquet")
    matching_files.extend(glob.glob(pattern_60hz))

    if not matching_files:
        print(f"No parquet files found for clip {video_number}")
        return

    print(f"Found {len(matching_files)} files for clip {video_number}")

    # Separate files into training (users 2-9) and testing (users 10-11)
    training_files = []
    testing_files = []

    for file_path in matching_files:
        # Extract user number from filename
        match = re.search(r'user(\d+)_clip', os.path.basename(file_path))
        if match:
            user_id = int(match.group(1))
            if 2 <= user_id <= 9:
                training_files.append((file_path, user_id))
            elif 10 <= user_id <= 11:
                testing_files.append((file_path, user_id))

    # for simplicity of splitting 80/20, we use users 2-9 for training, 10 & 11 for testing
    print(f"Training files: {len(training_files)} (users 2-9)")
    print(f"Testing files: {len(testing_files)} (users 10-11)")

    # Build index DataFrames for training and testing
    def build_index_df(file_list, semantic_path):
        rows = []
        for csv_path, user_id in file_list:
            # Load the parquet file to get the number of frames
            df = pd.read_parquet(csv_path)
            num_frames = len(df)
            
            # Create entries for each frame (starting from seq_len to have enough history)
            for frame_idx in range(seq_len, num_frames):
                rows.append({
                    "user_id": user_id,
                    "csv_path": csv_path,
                    "frame_idx": frame_idx,
                    "semantic_path": semantic_path
                })

        return pd.DataFrame(rows)

    # Build training and testing index DataFrames
    print("\nBuilding training index...")
    training_index_df = build_index_df(training_files, semantic_path)
    print(f"Training samples: {len(training_index_df)}")

    print("\nBuilding testing index...")
    testing_index_df = build_index_df(testing_files, semantic_path)
    print(f"Testing samples: {len(testing_index_df)}")

    # Initialize datasets
    training_dataset = HeadMotionDataset(index_df=training_index_df, seq_len=seq_len)
    testing_dataset = HeadMotionDataset(index_df=testing_index_df, seq_len=seq_len)

    print(f"\nTraining dataset size: {len(training_dataset)}")
    print(f"Testing dataset size: {len(testing_dataset)}")

    # Sample from training dataset
    if len(training_dataset) > 0:
        print("\n--- Training Sample ---")
        train_sample = training_dataset[0]
        print("Motion Sequence shape:", train_sample["motion_seq"].shape)
        print("Semantic Tensor shape:", train_sample["semantic"].shape)
        print("User ID:", train_sample["user_id"].item())
        print("Target shape:", train_sample["target"].shape)
        print("First 5 values of Motion Sequence:")
        print(train_sample["motion_seq"][:5])

    # Sample from testing dataset
    if len(testing_dataset) > 0:
        print("\n--- Testing Sample ---")
        test_sample = testing_dataset[0]
        print("Motion Sequence shape:", test_sample["motion_seq"].shape)
        print("Semantic Tensor shape:", test_sample["semantic"].shape)
        print("User ID:", test_sample["user_id"].item())
        print("Target shape:", test_sample["target"].shape)
        print("First 5 values of Motion Sequence:")
        print(test_sample["motion_seq"][:5])


if __name__ == "__main__":
    main()