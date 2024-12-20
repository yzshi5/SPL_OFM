import torch
import matplotlib.pyplot as plt
import sys
#from neuralop.models import FNO as _FNO

from .Diff_FNO_base import Diff_FNO as _Diff_FNO
#from Diff_FNO_base import Diff_FNO as 

"""
 A version of the time-conditioned FNO model.
 Uses the new neuralop package.
 Works by concatenating time as an input channel.
"""

def t_allhot(t, shape):
    batch_size = shape[0]
    n_channels = shape[1]
    dim = shape[2:]
    n_dim = len(dim)

    t = t.view(batch_size, *[1]*(n_channels+n_dim))
    t = t * torch.ones(batch_size, 1, *dim, device=t.device)
    return t


class Diff_FNO(torch.nn.Module):
    def __init__(self, modes, vis_channels, hidden_channels, proj_channels, x_dim=1, t_scaling=1000):
        super(Diff_FNO, self).__init__()
        
        self.t_scaling = t_scaling
        
        # modes = 16
        # hidden = 32
        # proj = 64
        
        #model = TFNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
        n_modes = (modes, ) * x_dim   # Same number of modes in each x dimension
        in_channels = vis_channels + x_dim + 1  # visual channels + spatial embedding + time embedding

        self.model = _Diff_FNO(n_modes=n_modes, 
                         hidden_channels=hidden_channels, projection_channels=proj_channels,
                         in_channels=in_channels, out_channels=vis_channels)
        
        
    def forward(self, t, u):
        # u: (batch_size, channels, h, w)
        # t: either scalar or (batch_size,)

        t = t / self.t_scaling
        #print("t in the model:{}".format(t))
        batch_size = u.shape[0]
        dims = u.shape[2:]
        
        if t.dim() == 0 or t.numel() == 1:
            t = torch.ones(u.shape[0], device=t.device) * t

        assert t.dim() == 1
        assert t.shape[0] == u.shape[0]

        # Concatenate time as a new channel
        t = t_allhot(t, u.shape)
        #print('t max:{}, t min:{}'.format(t.max(), t.min()))
        # Concatenate position as new channel(s)
        posn_emb = make_posn_embed(batch_size, dims).to(u.device)
        u = torch.cat((u, posn_emb, t), dim=1).float() # todo fix precision
        
        out = self.model(u)

        return out
    

def make_posn_embed(batch_size, dims):
    
    if len(dims) == 1:
        # Single channel of spatial embeddings
        emb = torch.linspace(0, 1, dims[0])
        emb = emb.unsqueeze(0).repeat(batch_size, 1, 1)
    elif len(dims) == 2:
        # 2 Channels of spatial embeddings
        x1 = torch.linspace(0, 1, dims[1]).repeat(dims[0], 1).unsqueeze(0)
        x2 = torch.linspace(0, 1, dims[0]).repeat(dims[1], 1).T.unsqueeze(0)
        emb = torch.cat((x1, x2), dim=0)  # (2, dims[0], dims[1])

        # Repeat along new batch channel
        emb = emb.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (batch_size, 2, *dims)
    else:
        raise NotImplementedError
    
    
    return emb
    

class FNO1dgano(torch.nn.Module):
    def __init__(self, modes, hidden_channels, proj_channels):
        super(FNO1dgano, self).__init__()
        
        #model = TFNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
        in_channels = 2  # visual channels + spatial embedding + time embedding

        self.model = _FNO(n_modes=(modes,), 
                         hidden_channels=hidden_channels, projection_channels=proj_channels,
                         in_channels=in_channels, out_channels=1)
        
        
    def forward(self, u):
        # u: (batch_size, n_x, channels)
        batch_size, n_x = u.squeeze().shape
        
        u = u.reshape(batch_size, 1, n_x)
        # print(u.shape)
        posn_emb = make_posn_embed(batch_size, [n_x]).to(u.device)
        u = torch.cat((u, posn_emb), dim=1).float() # todo fix precision
        # print(u.shape)
        out = self.model(u)

        return out