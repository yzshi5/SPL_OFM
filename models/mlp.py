import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """ Very simple MLP with one hidden layer.
    Handles time via concatenation. 
    """

    def __init__(self, dim_in, dim_hidden):
        super().__init__()

        self.act = nn.GELU()

        self.layers = nn.Sequential(
            nn.Linear(dim_in + 1, dim_hidden),
            self.act,
            nn.Linear(dim_hidden, dim_hidden),
            self.act,
            nn.Linear(dim_hidden, dim_hidden),
            self.act,
            nn.Linear(dim_hidden, dim_hidden),
            self.act,
            nn.Linear(dim_hidden, dim_hidden),
            self.act,
            nn.Linear(dim_hidden, dim_in),
        )

    def forward(self, t, x):
        # t: [batch_size, 1]
        # x: [batch_size, dim_in]
        x = x.type(torch.float32)

        if t.dim() == 0:
            t = torch.ones(x.shape[0], 1) * t

        model_in = torch.hstack([t, x])
        model_out = self.layers(model_in)
        return model_out
