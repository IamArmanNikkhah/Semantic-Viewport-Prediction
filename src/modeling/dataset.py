from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


class HeadMotionDataset(Dataset):
    """
    Phase 2 – Dataset class.

    For each index row, this returns:
      - motion_seq: last `seq_len` timesteps of (yaw, pitch) before frame_idx
      - semantic:  Phase 1 P_sem tensor for frame_idx
      - user_id:   scalar user index for nn.Embedding(Num_Users, 11)
      - target:    yaw, pitch at frame_idx (what we want to predict)
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        seq_len: int = 30,
        yaw_col: str = "yaw",
        pitch_col: str = "pitch",
    ):

        super().__init__()

        # Store a copy of the metadata index so __getitem__ can use row numbers.
        self.index_df = index_df.reset_index(drop=True)

        # Number of timesteps of motion history.
        self.seq_len = seq_len

        # Remember which CSV columns contain yaw & pitch.
        self.yaw_col = yaw_col
        self.pitch_col = pitch_col

    def __len__(self) -> int:
        
        return len(self.index_df)

    def __getitem__(self, idx: int) -> dict:
        
        # 1. Read metadata row
        row = self.index_df.iloc[idx]

        # user_id: which user this sample belongs to
        user_id = int(row["user_id"])

        # csv_path: location of the Week 1 motion file for this sample
        csv_path = row["csv_path"]

        # frame_idx: index of the CURRENT timestamp in that CSV
        frame_idx = int(row["frame_idx"])

        # semantic_path: where the Phase 1 semantic tensor was saved
        semantic_path = row["semantic_path"]

        # 2. Load motion CSV
        df = pd.read_parquet(csv_path)

        # make sure the requested frame index exists.
        if frame_idx < 0 or frame_idx >= len(df):
            raise IndexError(
                f"frame_idx {frame_idx} out of bounds for file {csv_path} "
                f"(num rows = {len(df)})"
            )

        # 3. Slice motion history
        history_end = frame_idx              # exclusive end index
        history_start = max(0, history_end - self.seq_len)

        # extract just the yaw & pitch columns for the history window.
        # .values → numpy array with shape [num_history_rows, 2]
        history_np = df[[self.yaw_col, self.pitch_col]].iloc[
            history_start:history_end
        ].values
        if len(history_np) < self.seq_len:
            # Number of missing timesteps at the beginning
            pad_len = self.seq_len - len(history_np)

            # Create padding tensor [pad_len, 2] filled with zeros.
            pad_tensor = torch.zeros(pad_len, 2, dtype=torch.float32)

            # Convert available history rows to a tensor.
            history_tensor = torch.tensor(history_np, dtype=torch.float32)

            # Concatenate padding (older) + actual history (newer).
            motion_seq = torch.cat([pad_tensor, history_tensor], dim=0)
        else:
            # We have at least seq_len rows; take the **last** seq_len of them.
            history_np = history_np[-self.seq_len :]
            motion_seq = torch.tensor(history_np, dtype=torch.float32)

        # 4. Load semantic tensor
        semantic_tensor = torch.load(semantic_path)

        # Ensure it is a float tensor so it can go into nn.Linear etc.
        semantic_tensor = semantic_tensor.float()

        # 5. Build prediction target
        target_row = df[[self.yaw_col, self.pitch_col]].iloc[frame_idx].values
        target = torch.tensor(target_row, dtype=torch.float32)

        # 6. Pack everything into a dict
        sample = {
            # Component A input: motion encoder
            "motion_seq": motion_seq,  # [seq_len, 2]

            # Component B input: personalized semantic attention
            "semantic": semantic_tensor,  # [11, 4, 6]

            # User index for nn.Embedding(Num_Users, 11)
            "user_id": torch.tensor(user_id, dtype=torch.long),

            # Component C target: [next_yaw, next_pitch]
            "target": target,  # [2]
        }

        return sample