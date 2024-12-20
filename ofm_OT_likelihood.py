import numpy as np
import torch
from torchdiffeq import odeint
from util.true_gaussian_process import true_GPPrior
from util.util import make_grid, reshape_for_batchwise, plot_loss_curve, plot_samples
from torchcfm.optimal_transport import OTPlanSampler

import time

"""
Code modified from https://github.com/GavinKerrigan/functional_flow_matching
"""
"""
key elements
(1) GP prior with covariance calculated with cholesky decomposition 
(2) optimal transport plan
(3) likelihood estimation for batch data
(3) precise likelihood estimation for a sample
(4) codomain likelihood estimation for a sample
"""

class OFMModel:
    def __init__(self, model, kernel_length=0.001, kernel_variance=1.0, nu=0.5, sigma_min=1e-4, device='cpu', dtype=torch.double, dims=None):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.gp = true_GPPrior(lengthscale=kernel_length, var=kernel_variance, nu=nu, device=device, dims=dims)
        self.ot_sampler = OTPlanSampler(method="exact")
        self.sigma_min = sigma_min

    def sample_gp_noise(self, x_data):
        # sample GP noise with OT 
        
        batch_size = x_data.shape[0]
        n_channels = x_data.shape[1]
        dims = x_data.shape[2:]
        n_dims = len(dims)
        
        # Sample from prior GP
        query_points = make_grid(dims)
        
        # GP noise : [batch_size, n_channels, *dims]
        #x_0 = self.gp.sample_from_prior(query_points, dims, n_samples=batch_size, n_channels=n_channels) 
        x_0 = self.gp.sample_from_prior(dims, n_samples=batch_size, n_channels=n_channels) 
        x_0, x_data = self.ot_sampler.sample_plan(x_0, x_data)
        
        return x_0, x_data
        
    def simulate(self, t, x_0, x_data):
        # t: [batch_size,]
        # x_data: [batch_size, n_channels, *dims]
        # samples from p_t(x | x_data)
        
        batch_size = x_data.shape[0]
        n_channels = x_data.shape[1]
        dims = x_data.shape[2:]
        n_dims = len(dims)
        
        # Sample from prior GP
        noise = self.gp.sample_from_prior(dims, n_samples=batch_size, n_channels=n_channels)
    
        # Construct mean/variance parameters
        t = reshape_for_batchwise(t, 1 + n_dims)
        
        mu = t * x_data + (1 - t) * x_0
        samples = mu + self.sigma_min * noise

        assert samples.shape == x_data.shape
        return samples
    
    def get_conditional_fields(self, x0, x1):
        # computes v_t(x_noisy | x_data)
        # x_data, x_noisy: (batch_size, n_channels, *dims)

        return x1 - x0

    def train(self, train_loader, optimizer, epochs, 
                scheduler=None, test_loader=None, eval_int=0, 
                save_int=0, generate=False, save_path=None, saved_model=False):

        tr_losses = []
        te_losses = []
        eval_eps = []
        evaluate = (eval_int > 0) and (test_loader is not None)

        model = self.model
        device = self.device
        dtype = self.dtype

        first = True
        for ep in range(1, epochs+1):
            ##### TRAINING LOOP
            t0 = time.time()
            model.train()
            tr_loss = 0.0

            for batch in train_loader:
                batch = batch.to(device)
                batch_size = batch.shape[0]

                if first:
                    self.n_channels = batch.shape[1]
                    self.train_dims = batch.shape[2:]
                    first = False
                    
                # GP noise with OT reorder
                x_0, x_data = self.sample_gp_noise(batch)
        
                # t ~ Unif[0, 1)
                t = torch.rand(batch_size, device=device)
                # Simluate p_t(x | x_1)
                x_t = self.simulate(t, x_0, x_data)
                # Get conditional vector fields
                target = self.get_conditional_fields(x_0, x_data)

                x_t = x_t.to(device)
                target = target.to(device)         

                # Get model output
                #print('t before the model :{}'.format(t))
                model_out = model(t, x_t)

                # Evaluate loss and do gradient step
                optimizer.zero_grad()
                loss = torch.mean((model_out - target)**2 ) 
                loss.backward()
                optimizer.step()

                tr_loss += loss.item()

            tr_loss /= len(train_loader)
            tr_losses.append(tr_loss)
            if scheduler: scheduler.step()


            t1 = time.time()
            epoch_time = t1 - t0
            print(f'tr @ epoch {ep}/{epochs} | Loss {tr_loss:.6f} | {epoch_time:.2f} (s)')

            ##### EVAL LOOP
            if eval_int > 0 and (ep % eval_int == 0):
                t0 = time.time()
                eval_eps.append(ep)

                with torch.no_grad():
                    model.eval()

                    if evaluate:
                        te_loss = 0.0
                        for batch in test_loader:
                            batch = batch.to(device)
                            batch_size = batch.shape[0]

                            # GP noise with OT reorder
                            x_0, x_data = self.sample_gp_noise(batch)

                            # t ~ Unif[0, 1)
                            t = torch.rand(batch_size, device=device)
                            # Simluate p_t(x | x_1)
                            x_t = self.simulate(t, x_0, x_data)
                            # Get conditional vector fields
                            target = self.get_conditional_fields(x_0, x_data)

                            x_t = x_t.to(device)
                            target = target.to(device)  
                
                            model_out = model(t, x_t)

                            loss = torch.mean( (model_out - target)**2 )

                            te_loss += loss.item()

                        te_loss /= len(test_loader)
                        te_losses.append(te_loss)

                        t1 = time.time()
                        epoch_time = t1 - t0
                        print(f'te @ epoch {ep}/{epochs} | Loss {te_loss:.6f} | {epoch_time:.2f} (s)')


                    # genereate samples during training?
                    if generate:
                        samples = self.sample(self.train_dims, n_channels=self.n_channels, n_samples=16)
                        plot_samples(samples, save_path / f'samples_epoch{ep}.pdf')


            ##### BOOKKEEPING
            if saved_model == True:
                if ep % save_int == 0:
                    torch.save(model.state_dict(), save_path / f'epoch_{ep}.pt')

            if evaluate:
                plot_loss_curve(tr_losses, save_path / 'loss.pdf', te_loss=te_losses, te_epochs=eval_eps)
            else:
                plot_loss_curve(tr_losses, save_path / 'loss.pdf')


    @torch.no_grad()
    def sample(self, dims, n_channels=1, n_samples=1, n_eval=2, return_path=False, rtol=1e-5, atol=1e-5, method = 'dopri5'):
        # n_eval: how many timesteps in [0, 1] to evaluate. Should be >= 2. 
        # dims: dimensionality of domain, e.g. [64, 64] for 64x64 images

        t = torch.linspace(0, 1, n_eval, device=self.device)
        #grid = make_grid(dims)
        #x0 = self.gp.sample(grid, dims, n_samples=n_samples, n_channels=n_channels)
        x0 = self.gp.sample(dims, n_samples=n_samples, n_channels=n_channels)
        
        out = odeint(self.model, x0, t, method=method, rtol=rtol, atol=atol)

        if return_path:
            return out
        else:
            return out[-1]
    
    @torch.no_grad()
    def inv_sample(self, samples, n_eval=2, return_path=False, rtol=1e-5, atol=1e-5, forward=False, method='dopri5'):
        
        if forward == False:
            t = torch.linspace(1, 0, n_eval, device=self.device)
        else:
            t = torch.linspace(0, 1, n_eval, device=self.device)
            
        x0 = samples.to(self.device)

        out = odeint(self.model, x0, t, method=method, rtol=rtol, atol=atol)

        if return_path:
            return out
        else:
            return out[-1]        
 
        
    #def likelihood_fn(self, sample, rtol=1e-5, atol=1e-5, forward=False)
    #@torch.no_grad()
    
    

    def data_likelihood(self, samples, n_eval=2, rtol=1e-5, atol=1e-5, forward=False, hutchinson_type='Rademacher', retain_graph=True, method='dopri5'):
        ## samples : [batch_size=1, 1, *dims] 

        if hutchinson_type == 'Gaussian':
            noise = torch.randn_like(samples)
        elif hutchinson_type == 'Rademacher':
            noise = torch.randint_like(samples, low=0, high=2).float() * 2 - 1.
        else:
            raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")
            
        #noise = torch.randn_like(samples) #already to device), use this noise for all steps 
        shape = samples.shape
                
        
        def ode_func(t, x_t):
            ## return the the dirft and logp_grad, t is a scalar like [5]
            # check the t shape 
            
            sample = x_t[:-shape[0]].reshape(shape)
            
            with torch.enable_grad():
                sample.requires_grad_(True)
                vect_t = torch.ones(sample.shape[0], device=sample.device) * t
                drift = self.model(vect_t, sample) # [batch_size, n_channel, dim*]
                
                # calculate the logp with diveregence
                fn_eps = torch.sum(drift * noise)
                grad_fn_eps = torch.autograd.grad(fn_eps, sample, retain_graph=retain_graph)[0]
                #sample.requires_grad_(False)
                
                logp_grad = torch.sum(grad_fn_eps * noise, dim=tuple(range(1, len(shape))))              
            
            return torch.cat([torch.flatten(drift), logp_grad], dim=0)
                

        #t = torch.linspace(1, 0, n_eval, device=self.device)  
        if forward == False:
            t = torch.linspace(1, 0, n_eval, device=self.device)
        else:
            t = torch.linspace(0, 1, n_eval, device=self.device)
            
            
        #x0 = samples.to(self.device)
        x0 = torch.cat([torch.flatten(samples), torch.zeros(shape[0]).to(self.device)], dim=0)
        out = odeint(ode_func, x0, t, method=method, rtol=rtol, atol=atol)
        
        # out : [n_eval, -1], first part is the data, last part is the logp
        out_samples = out[-1, :-shape[0]].reshape(shape[0],-1)
        
        #print("out_smaples.shape:{}".format(out_samples.shape))
        
        if forward == False:
            prior_logp = self.gp.prior_likelihood(out_samples)
            out_logp = -out[-1,-shape[0]:] + prior_logp
            
        else:  #From sample to noise 
            prior_logp = self.gp.prior_likelihood(samples)
            out_logp = out[-1,-shape[0]:] + prior_logp
        
        return out_samples.reshape(shape), out_logp, prior_logp     
    

    def data_likelihood_precise(self, samples, n_eval=2, rtol=1e-5, atol=1e-5, forward=False,hutchinson_type='Rademacher', retain_graph=True, n_repeat=32, method='dopri5'):
        """
        retain_graph = True for regression, set as False for 
        """
        repeat_dim = (n_repeat,) + (1,) * (samples.dim() - 1) # [batch_size, 1...]
        samples = samples.repeat(repeat_dim)
        
        if hutchinson_type == 'Gaussian':
            noise = torch.randn_like(samples)
        elif hutchinson_type == 'Rademacher':
            noise = torch.randint_like(samples, low=0, high=2).float() * 2 - 1.
        else:
            raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")
            
        #noise = torch.randn_like(samples) #already to device), use this noise for all steps 
        shape = samples.shape
                
        
        def ode_func(t, x_t):
            ## return the the dirft and logp_grad, t is a scalar like [5]
            # check the t shape 
            
            sample = x_t[:-shape[0]].reshape(shape)
            
            with torch.enable_grad():
                sample.requires_grad_(True)
                vect_t = torch.ones(sample.shape[0], device=sample.device) * t
                drift = self.model(vect_t, sample) # [batch_size, n_channel, dim*]
                
                # calculate the logp with diveregence
                fn_eps = torch.sum(drift * noise)
                grad_fn_eps = torch.autograd.grad(fn_eps, sample, retain_graph=retain_graph)[0] 
                #sample.requires_grad_(False)
                
                logp_grad = torch.sum(grad_fn_eps * noise, dim=tuple(range(1, len(shape))))              
            
            return torch.cat([torch.flatten(drift), logp_grad], dim=0)
                

        #t = torch.linspace(1, 0, n_eval, device=self.device)  
        if forward == False:
            t = torch.linspace(1, 0, n_eval, device=self.device)
        else:
            t = torch.linspace(0, 1, n_eval, device=self.device)
            
            
        #x0 = samples.to(self.device)
        x0 = torch.cat([torch.flatten(samples), torch.zeros(shape[0]).to(self.device)], dim=0)
        out = odeint(ode_func, x0, t, method=method, rtol=rtol, atol=atol)
        
        # out : [n_eval, -1], first part is the data, last part is the logp
        out_samples = out[-1, :-shape[0]].reshape(shape[0],-1)
        
        #print("out_smaples.shape:{}".format(out_samples.shape))
        
        if forward == False:
            prior_logp = self.gp.prior_likelihood(out_samples)
            out_logp = -out[-1,-shape[0]:] + prior_logp
            
        else:  #From sample to noise 
            prior_logp = self.gp.prior_likelihood(samples)
            out_logp = out[-1,-shape[0]:] + prior_logp
        
        return out_samples.reshape(shape).mean(0), out_logp.mean(0), prior_logp.mean(0)     

    def data_likelihood_precise_codomain(self, samples, n_channels=1, n_eval=2, rtol=1e-5, atol=1e-5, forward=False,hutchinson_type='Rademacher', retain_graph=True, n_repeat=32, method='dopri5'):
        """
        retain_graph = True for regression, set as False for 
        """
        repeat_dim = (n_repeat,) + (1,) * (samples.dim() - 1) # [batch_size, 1...]
        samples = samples.repeat(repeat_dim)
        
        if hutchinson_type == 'Gaussian':
            noise = torch.randn_like(samples)
        elif hutchinson_type == 'Rademacher':
            noise = torch.randint_like(samples, low=0, high=2).float() * 2 - 1.
        else:
            raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")
            
        #noise = torch.randn_like(samples) #already to device), use this noise for all steps 
        shape = samples.shape
                
        
        def ode_func(t, x_t):
            ## return the the dirft and logp_grad, t is a scalar like [5]
            # check the t shape 
            
            sample = x_t[:-shape[0]].reshape(shape)
            
            with torch.enable_grad():
                sample.requires_grad_(True)
                vect_t = torch.ones(sample.shape[0], device=sample.device) * t
                drift = self.model(vect_t, sample) # [batch_size, n_channel, dim*]
                
                # calculate the logp with diveregence
                fn_eps = torch.sum(drift * noise)
                grad_fn_eps = torch.autograd.grad(fn_eps, sample, retain_graph=retain_graph)[0] 
                #sample.requires_grad_(False)
                
                logp_grad = torch.sum(grad_fn_eps * noise, dim=tuple(range(1, len(shape))))              
            
            return torch.cat([torch.flatten(drift), logp_grad], dim=0)
                

        #t = torch.linspace(1, 0, n_eval, device=self.device)  
        if forward == False:
            t = torch.linspace(1, 0, n_eval, device=self.device)
        else:
            t = torch.linspace(0, 1, n_eval, device=self.device)
            
            
        #x0 = samples.to(self.device)
        x0 = torch.cat([torch.flatten(samples), torch.zeros(shape[0]).to(self.device)], dim=0)
        out = odeint(ode_func, x0, t, method=method, rtol=rtol, atol=atol)
        
        # out : [n_eval, -1], first part is the data, last part is the logp
        out_samples = out[-1, :-shape[0]].reshape(shape[0],-1)
        
        #print("out_smaples.shape:{}".format(out_samples.shape))
        
        if forward == False:

            prior_logp = self.gp.prior_likelihood_codomain(out_samples, n_channels)
            out_logp = -out[-1,-shape[0]:] + prior_logp
            
        else:  #From sample to noise 
            prior_logp = self.gp.prior_likelihood_codomain(samples, n_channels)
            out_logp = out[-1,-shape[0]:] + prior_logp
        
        return out_samples.reshape(shape).mean(0), out_logp.mean(0), prior_logp.mean(0)     
