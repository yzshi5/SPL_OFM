import torch
import numpy as np
import matplotlib.pyplot as plt
from util.util2 import MatReader


def make_grid(dims, x_min=0, x_max=1):
    """ Creates a 1D or 2D grid based on the list of dimensions in dims.

    Example: dims = [64, 64] returns a grid of shape (64*64, 2)
    Example: dims = [100] returns a grid of shape (100, 1)
    """
    if len(dims) == 1:
        grid = torch.linspace(x_min, x_max, dims[0])
        grid = grid.unsqueeze(-1)
    elif len(dims) == 2:
        _, _, grid = make_2d_grid(dims)   
    elif len(dims) == 3: # Shouldn't try it, too large
        _, _, _, grid = make_3d_grid(dims)
        
    return grid


def make_2d_grid(dims, x_min=0, x_max=1):
    # Makes a 2D grid in the format of (n_grid, 2)
    x1 = torch.linspace(x_min, x_max, dims[0])
    x2 = torch.linspace(x_min, x_max, dims[1])
    x1, x2 = torch.meshgrid(x1, x2, indexing='ij')
    grid = torch.cat((
        x1.contiguous().view(x1.numel(), 1),
        x2.contiguous().view(x2.numel(), 1)),
        dim=1)
    return x1, x2, grid

def make_3d_grid(dims, x_min=0, x_max=1):
    x1 = torch.linspace(x_min, x_max, dims[0])
    x2 = torch.linspace(x_min, x_max, dims[1])
    x3 = torch.linspace(x_min, x_max, dims[2])
    x1, x2, x3 = torch.meshgrid(x1, x2, x3, indexing='ij')
    grid = torch.cat((
        x1.contiguous().view(x1.numel(), 1),
        x2.contiguous().view(x2.numel(), 1),
        x3.contiguous().view(x3.numel(), 1)),
        dim=1)    
    return x1, x2, x3, grid
    
def reshape_for_batchwise(x, k):
        # need to do some ugly shape-hacking here to get appropriate number of dims
        # maps tensor (n,) to (n, 1, 1, ..., 1) where there are k 1's
        return x.view(-1, *[1]*k)

def reshape_channel_last(x):
    # maps a tensor (B, C, *dims) to (B, *dims, C)
    k = x.ndim
    idx = list(range(k))
    idx.append(idx.pop(1))
    return x.permute(idx)

def reshape_channel_first(x):
    # maps a tensor (B, *dims, C) to (B, C, *dims)
    k = x.ndim
    idx = list(range(k))
    idx.insert(1, idx.pop())
    return x.permute(idx)


def load_navier_stokes(path=None, shuffle=True):
    if not path:
        path = '../data/NavierStokes_V1e-3_N5000_T50/ns_V1e-3_N5000_T50.mat'
    r = MatReader(path)
    r._load_file()
    u = r.read_field('u')  # (5k, 64, 64, 50)
     
    u = u.permute(0, -1, 1, 2).reshape(-1, 64, 64).unsqueeze(1)  # (25k, 1, 64, 64)
    
    if shuffle:
        idx = torch.randperm(u.shape[0])
        u = u[idx]
        
    return u


def plot_loss_curve(tr_loss, save_path, te_loss=None, te_epochs=None, logscale=True):
    fig, ax = plt.subplots()

    if logscale:
        ax.semilogy(tr_loss, label='tr')
    else:
        ax.plot(tr_loss, label='tr')
    if te_loss is not None:
        te_epochs = np.asarray(te_epochs)
        if logscale:
            ax.semilogy(te_epochs-1, te_loss, label='te')  # assume te_epochs is 1-indexed
        else:
            ax.plot(te_epochs-1, te_loss, label='te')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='upper right')

    plt.savefig(save_path)
    plt.close(fig)


def plot_samples(samples, save_path):
    n = samples.shape[0]
    sqrt_n = int(np.sqrt(n))

    fig, axs = plt.subplots(sqrt_n, sqrt_n, figsize=(8,8))

    samples = samples.permute(0, 2, 3, 1)  # (b, c, h, w) --> (b, h, w, c)
    samples = samples.detach().cpu()

    for i in range(n):
        j, k = i//sqrt_n, i%sqrt_n
        
        axs[j, k].imshow(samples[i])
        
        axs[j, k].set_xticks([])
        axs[j, k].set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.savefig(save_path)
    plt.close(fig)


def sample_many(wrapper, n_samples, dims, n_channels=1, batch_size=500, save_path=None):
    n_batches = n_samples // batch_size
    n_samples = n_batches * batch_size
    print(f'Generating {n_samples} samples')


    samples = []
    generated = 0
    while generated < n_samples:
        print(f'... generated {generated}/{n_samples}')
        try:
            sample = wrapper.sample(dims, n_samples=batch_size, n_channels=n_channels)
            samples.append(sample.detach().cpu())
            del sample
            torch.cuda.empty_cache()
            generated += batch_size
        except:
            print('NaN, retry')

    samples = torch.stack(samples)

    if save_path:
        torch.save(samples, save_path)
    return samples