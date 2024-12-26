import torch
import torch.nn as nn
import torch.nn.functional as F

class DiveBlock(nn.Module):
    def __init__(self,
                 d_embed,
                 d_hidden,
                 d_board):
        super().__init__()
        h_kernel = (2 * d_board[0] - 1, 1)
        h_padding = (d_board[0] - 1, 0)
        v_kernel = (1, 2 * d_board[1] - 1)
        v_padding = (0, d_board[1] - 1)
        self.conv_up_h = nn.Conv2d(d_embed, d_hidden, h_kernel, padding=h_padding)
        self.conv_up_v = nn.Conv2d(d_embed, d_hidden, v_kernel, padding=v_padding)
        self.conv_down_h = nn.Conv2d(d_hidden, d_embed, h_kernel, padding=h_padding)
        self.conv_down_v = nn.Conv2d(d_hidden, d_embed, v_kernel, padding=v_padding)

    def forward(self, x):
        y = self.conv_up_h(x) + self.conv_up_v(x)
        y = F.relu(y)
        y = self.conv_down_h(y) + self.conv_down_v(y)
        return x + y

class DiveModel(nn.Module):
    def __init__(self,
                 d_embed,
                 n_block = 2,
                 d_hidden = None,
                 d_board = (4, 4),                 
                 d_output = 4,
                 ):
        super().__init__()
        if not d_hidden:
            d_hidden = 4 * d_embed
        d_flatten = d_board[0] * d_board[1] * d_embed
        self.blocks = nn.Sequential(*[DiveBlock(d_embed, d_hidden, d_board) for i in range(n_block)])
        self.q_head = nn.Sequential(nn.Linear(d_flatten, d_flatten),
                                    nn.ReLU(),
                                    nn.Linear(d_flatten, d_output))
        
    def forward(self, x: torch.Tensor):
        # x: [B, M, N, C], y: [B, d_output = 4]
        x = x.permute(0, 3, 1, 2)
        x = self.blocks(x)
        y = x.flatten(-3, -1)
        y = self.q_head(y)
        return y