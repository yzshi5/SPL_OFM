import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import FNO as _FNO

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

    t = t.view(batch_size, *[1]*(1+n_dim))
    t = t * torch.ones(batch_size, 1, *dim, device=t.device)
    return t

def conds_allhot(conds, shape):
    batch_size = shape[0]
    dim = shape[2:]
    n_dim = len(dim)

    # expand conds
    
    if n_dim == 1:
        conds = conds.unsqueeze(-1)
    elif n_dim == 2:
        conds = conds.unsqueeze(-1).unsqueeze(-1)
    elif n_dim == 3:
        conds = conds.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
    conds = conds.repeat(1,1, *dim)
    return conds

class FNO(torch.nn.Module):
    def __init__(self, modes, vis_channels, hidden_channels, proj_channels, x_dim=1, t_scaling=1):
        super(FNO, self).__init__()
        
        self.t_scaling = t_scaling
        
        # modes = 16
        # hidden = 32
        # proj = 64
        
        #model = TFNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
        n_modes = (modes, ) * x_dim   # Same number of modes in each x dimension
        in_channels = vis_channels + x_dim + 1  # visual channels + spatial embedding + time embedding

        self.model = _FNO(n_modes=n_modes, 
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
    

class FNO_cond(torch.nn.Module):
    def __init__(self, modes, vis_channels, hidden_channels, proj_channels,conds_channels=1, x_dim=1, t_scaling=1):
        super(FNO_cond, self).__init__()
        
        self.t_scaling = t_scaling
        
        # modes = 16
        # hidden = 32
        # proj = 64
        # conds_channels, like the locs of hypocenter (x, y, z)
        
        #model = TFNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
        n_modes = (modes, ) * x_dim   # Same number of modes in each x dimension
        in_channels = vis_channels + x_dim + 1 + conds_channels  # visual channels + spatial embedding + time embedding + conds embedding

        self.model = _FNO(n_modes=n_modes, 
                         hidden_channels=hidden_channels, projection_channels=proj_channels,
                         in_channels=in_channels, out_channels=vis_channels)
        
        
    def forward(self, t, u, conds):
        # u: (batch_size, channels, h, w)
        # t: either scalar or (batch_size,)
        # conds : (batch_size, m) 

        t = t / self.t_scaling
        #print("t in the model:{}".format(t))
        batch_size = u.shape[0]
        dims = u.shape[2:]
        
        if t.dim() == 0 or t.numel() == 1:
            t = torch.ones(u.shape[0], device=t.device) * t

        assert t.dim() == 1
        assert t.shape[0] == u.shape[0]
        assert conds.dim() == 2

        # Concatenate time as a new channel
        t = t_allhot(t, u.shape)
        conds = conds_allhot(conds, u.shape)
        
        #print('t max:{}, t min:{}'.format(t.max(), t.min()))  
        # Concatenate position as new channel(s)
        posn_emb = make_posn_embed(batch_size, dims).to(u.device)
        u = torch.cat((u, posn_emb, t, conds), dim=1).float() # todo fix precision
        
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
        
   ## new  
    elif len(dims) == 3:
        x1 = torch.linspace(0, 1, dims[0]).reshape(1, dims[0], 1, 1).repeat(1, 1, dims[1], dims[2])
        x2 = torch.linspace(0, 1, dims[1]).reshape(1, 1, dims[1], 1).repeat(1, dims[0], 1, dims[2])
        x3 = torch.linspace(0, 1, dims[2]).reshape(1, 1, 1, dims[2]).repeat(1, dims[0], dims[1], 1)
        emb = torch.cat((x1, x2, x3), dim=0)
        
        emb = emb.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1) #(batch_size, 3, *dims)
  
    else:
        raise NotImplementedError
    
    
    return emb
    

