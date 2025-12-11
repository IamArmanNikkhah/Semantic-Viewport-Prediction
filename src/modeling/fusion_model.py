import torch
import torch.nn as nn
import torch.nn.functional as F

#Component A

def __init__( self,
        input_dim: int = 2, # yaw and pitch
        hidden_dim: int = 64,
        num_layers: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

#Takes [B(Batch size), T(Seq len),2 (yaw and pitch)] and feeds it into GRU to get the history in one vector and then it returns the last vector.
def forward(self, motion_seq: torch.Tensor) -> torch.Tensor:
        _, h_n = self.rnn(motion_seq)
        h_last = h_n[-1]
        return h_last

