import argparse
import glob
import os
import re

import pandas as pd
import torch
from src.modeling.dataset import HeadMotionDataset

"""
Script to create individual dataset tensors for each user/video combination.
This script processes standardized motion data and semantic priors,
generating combined tensors saved to data/combined/
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
    combined_dir = "data/combined"

    # Create combined directory if it doesn't exist
    os.makedirs(combined_dir, exist_ok=True)

    # Check if semantic prior exists
    if not os.path.exists(semantic_path):
        print(f"Error: Semantic prior not found at {semantic_path}")
        return

    # Find all 60hz parquet files matching the video number pattern
    pattern_60hz = os.path.join(standardized_dir, f"user*_clip{video_number}_60hz.parquet")
    matching_files = glob.glob(pattern_60hz)

    if not matching_files:
        print(f"No parquet files found for clip {video_number}")
        return

    print(f"Found {len(matching_files)} files for clip {video_number}")

    # Process each parquet file
    for file_path in matching_files:
        # Extract user number from filename
        match = re.search(r'user(\d+)_clip', os.path.basename(file_path))
        if not match:
            print(f"Skipping file with unexpected format: {file_path}")
            continue
        
        user_id = int(match.group(1))
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        print(f"\nProcessing {base_name}...")

        # Build index DataFrame for this single file
        rows = []
        df = pd.read_parquet(file_path)
        num_frames = len(df)
        
        # Create entries for each frame (starting from seq_len to have enough history)
        for frame_idx in range(seq_len, num_frames):
            rows.append({
                "user_id": user_id,
                "csv_path": file_path,
                "frame_idx": frame_idx,
                "semantic_path": semantic_path
            })

        if not rows:
            print(f"  No valid samples (file has {num_frames} frames, need at least {seq_len})")
            continue

        index_df = pd.DataFrame(rows)
        print(f"  Created {len(index_df)} samples")

        # Initialize dataset
        dataset = HeadMotionDataset(index_df=index_df, seq_len=seq_len)

        # Collect all samples into lists
        motion_seqs = []
        semantics = []
        user_ids = []
        targets = []

        for i in range(len(dataset)):
            sample = dataset[i]
            motion_seqs.append(sample["motion_seq"])
            semantics.append(sample["semantic"])
            user_ids.append(sample["user_id"])
            targets.append(sample["target"])

        # Stack into tensors
        combined_tensor = {
            "motion_seq": torch.stack(motion_seqs),      # [N, seq_len, 2]
            "semantic": torch.stack(semantics),          # [N, 11, 4, 6]
            "user_id": torch.stack(user_ids),            # [N]
            "target": torch.stack(targets),              # [N, 2]
        }

        # Save to data/combined as a single .pt for the fusion model to read from per user per vidoe
        output_path = os.path.join(combined_dir, f"{base_name}.pt")
        torch.save(combined_tensor, output_path)
        print(f"  Saved tensor to {output_path}")
        print(f"  Tensor shapes: motion_seq={combined_tensor['motion_seq'].shape}, "
              f"semantic={combined_tensor['semantic'].shape}, "
              f"user_id={combined_tensor['user_id'].shape}, "
              f"target={combined_tensor['target'].shape}")
        print(f"  First 5 motion sequences (first sample):")
        print(combined_tensor['motion_seq'][0, :5])

    print(f"\nProcessing complete! Tensors saved to {combined_dir}/")


if __name__ == "__main__":
    main()