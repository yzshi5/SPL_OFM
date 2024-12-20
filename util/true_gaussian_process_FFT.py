import torch
import numpy as np


class true_GPPrior(object):
    
    """ 
    FFT method for efficient 1D, 2D, 3D GRF implementation. (
    Wrapper around some torch utilities that makes prior sampling easy.
    """

    
    def __init__(self, lengthscale=None, dims=None, device='cpu', sigma_f=1.0):
        """
        kernel/mean/lengthscale/var: parameters of kernel
        you should choose right parameter to avoid numerical instability of the cov matrix
        """
        #assert len(dims) == 3 , 'not 3D' 
        
        if lengthscale == None:
            lengthscale = [0.05] * len(dims)
        
        self.lengthscale = lengthscale 
        self.dims = dims
        
        kernel_fft = self.new_dist(lengthscale, dims)
        self.kernel_fft = torch.sqrt(kernel_fft).unsqueeze(0).to(device) # shape [1, n_x, n_y, n_t] 
        self.device = device

    def new_dist(self, lengthscale, dims, sigma_f=1.0):

        correlation_lengths = np.array(lengthscale) * np.array(dims) 
        ranges = [torch.linspace(-s/2., s/2., steps=s) for s in dims]
        grid = torch.meshgrid(*ranges, indexing='ij')
        
        squared_distances = sum([(g / l)**2 for g, l in zip(grid, correlation_lengths)])
        # Compute the Gaussian kernel
        kernel = sigma_f**2 * torch.exp(-torch.sqrt(squared_distances))
        # FFT shift to align the kernel correctly for FFT use
        kernel = torch.fft.fftshift(kernel)
        kernel_fft = torch.fft.fftn(kernel)
        
        return kernel_fft
        
    def sample(self, dims, n_samples=1, n_channels=1):
        
        if len(dims) == 1:
            fft_dims = [1]
        elif len(dims) == 2:
            fft_dims = [1, 2]
        elif len(dims) == 3:
            fft_dims = [1, 2, 3]
        
        kernel_fft = self.new_dist(self.lengthscale, dims)
        kernel_fft = torch.sqrt(kernel_fft).unsqueeze(0).to(self.device) # shape [1, n_x, n_y, n_t] 
        
        noise = torch.randn([n_samples*n_channels, *dims]).to(self.device)
        noise_fft = torch.fft.fftn(noise, dim=fft_dims)
        grf_fft = kernel_fft * noise_fft
        
        samples = torch.fft.ifftn(grf_fft, dim=fft_dims).real
        
        samples = samples.reshape(n_samples, n_channels, *dims)
        
        return samples
    
    def sample_from_prior(self, n_samples=1, n_channels=1):
        """
        fixed prior
        """
        dims = self.dims
        
        if len(dims) == 1:
            fft_dims = [1]
        elif len(dims) == 2:
            fft_dims = [1, 2]
        elif len(dims) == 3:
            fft_dims = [1, 2, 3]
            
        noise = torch.randn([n_samples*n_channels, *dims]).to(self.device)
        noise_fft = torch.fft.fftn(noise, dim=fft_dims)
        grf_fft = self.kernel_fft * noise_fft
        
        samples = torch.fft.ifftn(grf_fft, dim=fft_dims).real
        
        samples = samples.reshape(n_samples, n_channels, *dims)
        
        return samples
                
    
        

        
    