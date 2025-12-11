import pandas as pd
from src.modeling.dataset import HeadMotionDataset

def main():
    # Example DataFrame to simulate the index_df
    data = {
        "user_id": [1],
        "csv_path": ["data/standardized/user2_clip10.parquet"],
        "frame_idx": [30],
        "semantic_path": ["data/semantic_priors/10.pt"]
    }
    index_df = pd.DataFrame(data)

    # Initialize the HeadMotionDataset
    dataset = HeadMotionDataset(index_df=index_df, seq_len=30)

    # Access the first item in the dataset
    sample = dataset[0]

    # Print the outputs
    print("Motion Sequence:", sample["motion_seq"].shape)
    print("Semantic Tensor:", sample["semantic"].shape)
    print("User ID:", sample["user_id"].item())
    print("Target:", sample["target"].shape)
    # Print the first 5 values of the motion sequence tensor
    print("First 5 values of Motion Sequence:", sample["motion_seq"][:5])

if __name__ == "__main__":
    main()