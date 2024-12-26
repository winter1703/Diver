import numpy as np
import torch
import torch.nn as nn

class DiveEmbed(nn.Module):
    def __init__(self, d_vocab, d_embed=32):
        super().__init__()
        self.embed = nn.Embedding(d_vocab, d_embed)

    def forward(self, x: torch.Tensor):
        # x: [B*, M, N] to y: [B*, M, N, C]
        x.to(self.device)
        return self.embed(x)
    
# TODO: Codes to study the embedding here