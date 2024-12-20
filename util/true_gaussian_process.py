import torch
from sklearn.gaussian_process.kernels import Matern
from util.util import make_grid
import numpy as np

def matern_kernel_cov(grids, length_scale, nu):
    """
    grids : [n_points, 1 or 2]
    """
    kernel = 1.0 * Matern(length_scale=length_scale, length_scale_bounds="fixed", nu=nu)
    return kernel(grids)


class true_GPPrior(torch.distributions.distribution.Distribution):
    
    """ Wrapper around some torch utilities that makes prior sampling easy.
    """

    def __init__(self, kernel=None, mean=None, lengthscale=None, var=None, nu=0.5, device='cpu', dims=None):
        """
        kernel/mean/lengthscale/var: parameters of kernel
        you should choose right parameter to avoid numerical instability of the cov matrix
        """
        assert var == 1, 'variance is not 1' 
        
        jitter = 1e-6
        ## kernel shape: [N, N], mean shape :[N]
        # dims should be 1D [n_x] or 2D [n_x, n_x]
        n_points = np.prod(dims)
        grids = make_grid(dims)
        matern_ker = matern_kernel_cov(grids, lengthscale, nu)
        
        self.lengthscale = lengthscale
        self.nu = nu
        self.dims = dims
        
        base_mu = torch.zeros(n_points).float()
        #add jitter 
        base_cov = torch.tensor(matern_ker).float() + jitter * torch.eye(matern_ker.shape[0])
        base_cov = base_cov.to(torch.float64) #can help improve numerical stability
        #add jitter 
        
        try:
            self.base_dist = torch.distributions.MultivariateNormal(base_mu.to(device), scale_tril=torch.linalg.cholesky_ex(base_cov.to(device))[0].to(torch.float32)) 
        except:
            self.base_dist = torch.distributions.MultivariateNormal(base_mu.to(device), scale_tril=torch.linalg.cholesky_ex(base_cov)[0].to(torch.float32).to(device)) #be careful of the numerical instability when calculating on GPU
            
        self.device = device

    def check_input(self, x, dims=None):
        assert x.ndim == 2, f'Input {x.shape} should have shape (n_points, dim)'
        if dims:
            assert x.shape[1] == len(dims), f'Input {x.shape} should have shape (n_points, dim)'

    def new_dist(self, dims):
        """ Creates a Normal distribution at the points in x.
        x: locations to query at, a flattened grid; tensor (n_points, dim)

        returns: a gpytorch distribution corresponding to a Gaussian at x
        """
        jitter = 1e-6        
        n_points = np.prod(dims)
        grids = make_grid(dims)
        matern_ker = matern_kernel_cov(grids, self.lengthscale, self.nu)
        
        base_mu = torch.zeros(n_points).float()
        base_cov = torch.tensor(matern_ker).float() + jitter * torch.eye(matern_ker.shape[0])
        base_cov = base_cov.to(torch.float64)        

        try:
            base_dist = torch.distributions.MultivariateNormal(base_mu.to(device), scale_tril=torch.linalg.cholesky_ex(base_cov.to(device))[0].to(torch.float32))       
        except:
            base_dist = torch.distributions.MultivariateNormal(base_mu.to(self.device), scale_tril=torch.linalg.cholesky_ex(base_cov)[0].to(torch.float32).to(self.device))  
            
        return base_dist
    
    def sample(self, dims, n_samples=1, n_channels=1):
        """ Draws samples from the GP prior.
        dims: list of dimensions of inputs; e.g. for a 64x64 input grid, dims=[64, 64]
        n_samples: number of samples to draw
        n_channels: number of independent channels to draw samples for

        returns: samples from the GP; tensor (n_samples, n_channels, dims[0], dims[1], ...)
        """
        
        #x = x.to(self.device)
        if dims == self.dims:
            distr = self.base_dist
        else:
            distr = self.new_dist(dims)
        samples = distr.sample(sample_shape = torch.Size([n_samples * n_channels, ]))
        samples = samples.reshape(n_samples, n_channels, *dims)
        
        return samples
        
    
    def sample_from_prior(self, dims, n_samples=1, n_channels=1):
        """
        fixed prior
        """
        samples = self.base_dist.sample(sample_shape = torch.Size([n_samples * n_channels, ]))
        samples = samples.reshape(n_samples, n_channels, *dims)
        
        return samples           
    
    def sample_train_data(self, dims, n_samples=1, n_channels=1, nbatch=1000):
        """
        calculation in cuda, but saved in cpu.
        iteratively 
        """
        samples_all = []

        sampled_num = 0
        nbatch = np.min([n_samples, nbatch])
              
        while sampled_num < n_samples:
            temp_sample = self.sample_from_prior(dims, nbatch).cpu()
            sampled_num += len(temp_sample)
            samples_all.append(temp_sample)
                
        samples_all = torch.vstack(samples_all)[:n_samples]
        return samples_all
        
    def prior_likelihood(self, x):
        """
        calculate the likelihood of the input.
        x shape:[n_batch, -1] 
        # only used in jacobian, already to(device), n_channels must be 1
        """
        x = torch.flatten(x, start_dim=1)
        logp = self.base_dist.log_prob(x)
        return logp
        
    ## for codomain data
    def prior_likelihood_codomain(self, x, n_channels=1):
        """
        calculate the likelihood of the input.
        x shape:[n_batch, -1] 
        # only used in jacobian, already to(device), n_channels must be 1
        """
        x = x.reshape(x.shape[0], n_channels, -1)
                                                         
        for i in range(n_channels):
            if i == 0:
                logp = self.base_dist.log_prob(torch.flatten(x[:,0],start_dim=1))
            else:
                logp += self.base_dist.log_prob(torch.flatten(x[:,i], start_dim=1))
        
        return logp    