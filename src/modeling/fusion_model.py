import torch
import torch.nn as nn


class FusionModel(nn.Module):
    
    def __init__(
        self,
        num_users: int,
        seq_len: int = 30,
        motion_input_size: int = 2,   # yaw, pitch
        motion_hidden_size: int = 64,
        semantic_channels: int = 11,  # number of semantic classes
        semantic_h: int = 4,
        semantic_w: int = 6,
        semantic_out_channels: int = 16,  # channels produced by semantic CNN
    ):
       
        super().__init__()

        # Component A: Motion Encoder (GRU over [T, 2])
        self.motion_encoder = nn.GRU(
            input_size=motion_input_size,
            hidden_size=motion_hidden_size,
            batch_first=True,  # input shape: [batch, seq_len, input_size]
        )

        # Component B: Personalized Semantic Attention

        self.user_preferences = nn.Embedding(num_users, semantic_channels)

        self.semantic_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=semantic_channels,
                out_channels=semantic_out_channels,
                kernel_size=3,
                padding=1,  # keep spatial size 4x6
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # → [batch, semantic_out_channels, 1, 1]
        )

        # Component C: Prediction Head
        fusion_input_dim = motion_hidden_size + semantic_out_channels

        self.pred_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # output: [next_yaw, next_pitch]
        )

    def forward(
        self,
        motion_seq: torch.Tensor,
        semantic_tensor: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:

        # Component A: Motion Encoder
        _, h_motion = self.motion_encoder(motion_seq)

        # Remove the layer dimension (1) → [batch, motion_hidden_size]
        h_motion = h_motion.squeeze(0)

        # Component B: Personalized Semantic Attention
        prefs = self.user_preferences(user_ids)
        prefs_expanded = prefs.unsqueeze(-1).unsqueeze(-1)
        weighted_semantic = semantic_tensor * prefs_expanded
        sem_features = self.semantic_cnn(weighted_semantic)
        h_semantic = sem_features.view(sem_features.size(0), -1)

        # Component C: Prediction Head

        h_fused = torch.cat([h_motion, h_semantic], dim=1)

        preds = self.pred_head(h_fused)  # [batch, 2]

        return preds
