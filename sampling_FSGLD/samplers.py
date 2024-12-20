import torch

from .precondSGLD import pSGLD
from .SGLD import SGLD
import copy
import numpy as np

class LangevinDynamics(object):
    """
    LangevinDynamics class for performing Langevin dynamics optimization.

    Args:
        x (torch.Tensor): Initial parameter values.
        func (callable): The loss function to be optimized.
        lr (float, optional): Initial learning rate. Default is 1e-2.
        lr_final (float, optional): Final learning rate. Default is 1e-4.
        max_itr (int, optional): Maximum number of iterations. Default is 1e4.
        device (str, optional): Device to perform computations on ('cpu' or
            'cuda'). Default is 'cpu'.

    Attributes:
        x (torch.Tensor): Current parameter values.
        optim (torch.optim.Optimizer): Optimizer for updating parameters.
        lr (float): Initial learning rate.
        lr_final (float): Final learning rate.
        max_itr (int): Maximum number of iterations.
        func (callable): The loss function.
        lr_fn (callable): Learning rate decay function.
        counter (float): Iteration counter.
    """

    def __init__(self,
                 x: torch.Tensor,
                 func: callable,
                 lr: float = 1e-2,
                 lr_final: float = 1e-4,
                 max_itr: int = 10000,
                 device: str = 'cpu',
                 pos_mask: bool = False,
                 length_scale : bool=False,
                 base_dist=None,
                 temperature=1,
                 momentum=0,
                 use_GP_noise=False):
        super(LangevinDynamics, self).__init__()
        print('update')
        self.x = x
        #self.x_noise = None
        print("Tempertaure:{}".format(temperature))
        self.optim = SGLD([self.x], lr, weight_decay=0.0, device=device, temperature=temperature, momentum=momentum)
            
        if (base_dist is not None) & (use_GP_noise==True):
            samples_all = []
            with torch.no_grad():
                for i in range(int(np.ceil(max_itr/10000))):
                    samples_all.append(base_dist.rsample(sample_shape=(10000,))[:,:,None].cpu())
            self.x_noise = torch.vstack(samples_all)
            self.base_dist = base_dist
        
        if use_GP_noise == False:
            print('start, white noise on A')
        else: 
            print('start, GP noise on A')
        self.lr = lr
        self.lr_final = lr_final
        self.max_itr = max_itr
        self.func = func
        self.lr_fn = self.decay_fn(lr=lr, lr_final=lr_final, max_itr=max_itr)
        self.counter = 0.0
        self.device = device
        self.use_GP_noise = use_GP_noise

    # optional for GP noise injection
    def sample_a_noise(self):
        return self.base_dist.rsample(sample_shape=(1,))[:,:,None]
        
    def sample(self, epoch) -> tuple:
        """
        Perform a Langevin dynamics step.

        Returns:
            tuple: A tuple containing the current parameter values and the loss
                value.
        """
        self.lr_decay()
        self.optim.zero_grad()
        loss = self.func(self.x)
        loss.backward()
        
        if self.use_GP_noise == False:
            self.optim.step()
        else:
            self.optim.step(input_noise=self.x_noise[epoch:epoch+1].to(self.device))
        #self.optim.step(input_noise=self.sample_a_noise())
        self.counter += 1
        return copy.deepcopy(self.x.data), loss.item()

    def decay_fn(self,
                 lr: float = 1e-2,
                 lr_final: float = 1e-4,
                 max_itr: int = 10000) -> callable:
        """
        Calculate the learning rate decay function.

        Args:
            lr (float): Initial learning rate.
            lr_final (float): Final learning rate.
            max_itr (int): Maximum number of iterations.

        Returns:
            callable: Learning rate decay function.
        """
        gamma = -0.55
        b = max_itr / ((lr_final / lr)**(1 / gamma) - 1.0)
        a = lr / (b**gamma)

        def lr_fn(t: float,
                  a: float = a,
                  b: float = b,
                  gamma: float = gamma) -> float:
            """
            Calculate the learning rate based on the iteration number.

            Args:
                t (float): Current iteration number.
                a (float): Scaling factor.
                b (float): Scaling factor.
                gamma (float): Exponent factor.

            Returns:
                float: Learning rate at the given iteration.
            """
            #print('t :{}, a :{}, b:{}, gamma:{}'.format(t,a,b,gamma))
            return a * ((b + t)**gamma)
        
        #print('lr_fn(0) :{}'.format(lr_fn(0)))
        return lr_fn

    def lr_decay(self):
        """
        Update the learning rate of the optimizer based on the current
        iteration.
        """
        for param_group in self.optim.param_groups:        
            param_group['lr'] = self.lr_fn(self.counter)
            #print('after lr : {}'.format(param_group['lr']))


class MetropolisAdjustedLangevin(object):
    """
    A class implementing the Metropolis-Adjusted Langevin algorithm.

    Args:
        x (torch.Tensor): Initial input tensor.
        func (callable): The target function to be minimized.
        lr (float, optional): Initial learning rate. Default is 1e-2.
        lr_final (float, optional): Final learning rate. Default is 1e-4.
        max_itr (float, optional): Maximum number of iterations. Default is
            1e4.
        device (str, optional): Device to perform computations on. Default is
            'cpu'.

    Attributes:
        x (list): List containing two tensors for current and proposed input
            values.
        loss (list): List containing two tensors for current and proposed loss
            values.
        grad (list): List containing two tensors for gradients of the loss with
            respect to input.
        optim (pSGLD): Custom optimizer instance.
        P (list): List containing two tensors representing proposal
            distributions.
        lr (float): Initial learning rate.
        lr_final (float): Final learning rate.
        max_itr (float): Maximum number of iterations.
        func (callable): Target function.
        lr_fn (callable): Learning rate decay function.
        counter (float): Iteration counter.
    """

    def __init__(self,
                 x: torch.Tensor,
                 func: callable,
                 lr: float = 1e-2,
                 lr_final: float = 1e-4,
                 max_itr: float = 1e4,
                 device: str = 'cpu',
                 pos_mask = None,
                 SGLD_type : bool=False):

        super(MetropolisAdjustedLangevin, self).__init__()

        self.x = [
            torch.zeros(x.shape, device=x.device, requires_grad=True),
            torch.zeros(x.shape, device=x.device, requires_grad=True)
        ]
        self.x[0].data = x.data.clone()
        self.x[1].data = x.data.clone()

        self.loss = [
            torch.zeros([1], device=x.device),
            torch.zeros([1], device=x.device)
        ]
        self.loss[0] = func(self.x[0])
        self.loss[1].data = self.loss[0].data

        self.grad = [
            torch.zeros(x.shape, device=x.device),
            torch.zeros(x.shape, device=x.device)
        ]
        self.grad[0].data = torch.autograd.grad(self.loss[0], [self.x[0]],
                                                create_graph=False)[0].data
        self.grad[1].data = self.grad[0].data

        if SGLD_type is True:
            self.optim = SGLD([self.x[1]], lr, weight_decay=0.0, device=device)
        else:
            self.optim = pSGLD([self.x[1]], lr, weight_decay=0.0, pos_mask=pos_mask)
            
        #self.optim = pSGLD([self.x[1]], lr, weight_decay=0.0)
        # P is 1 for SGLD, not for pSGLD

        self.lr = lr
        self.lr_final = lr_final
        self.max_itr = max_itr
        self.func = func
        self.lr_fn = self.decay_fn(lr=lr, lr_final=lr_final, max_itr=max_itr)
        self.counter = 0.0

    def sample(self) -> tuple:
        """
        Perform a Metropolis-Hastings step to generate a sample.

        Returns:
            tuple: A tuple containing the sampled input tensor and
                corresponding loss value.
        """
        accepted = False
        self.lr_decay()
        while not accepted:
            
            self.optim.zero_grad()
            loss = self.func(self.x[1])
            loss.backward()
            self.optim.step()
            self.loss[1].data = loss.data
        
            
            #self.x[1].grad = self.grad[1].data
            self.grad[1].data = self.x[1].grad
            #self.loss[1] = self.func(self.x[1])
            #self.grad[1].data = torch.autograd.grad(self.loss[1], [self.x[1]],
                                                    #create_graph=False)[0].data

            # something wrong with self.sample_prob
            alpha = min([1.0, self.sample_prob().cpu().detach()])
            print("alpha :{}, loss[0]:{}, loss[1]:{}".format(alpha, self.loss[0], self.loss[1]))
            
            if torch.rand([1]) <= alpha:
                self.grad[0].data = self.grad[1].data
                self.loss[0].data = self.loss[1].data
                self.x[0].data = self.x[1].data
                accepted = True
            else:
                self.x[1].data = self.x[0].data
                self.loss[1].data = self.loss[0].data
                
        self.counter += 1
        return copy.deepcopy(self.x[1].data), self.loss[1].item()

    def proposal_dist(self, idx: int) -> torch.Tensor:
        """
        Calculate the proposal distribution for Metropolis-Hastings.

        Args:
            idx (int): Index of the current tensor.

        Returns:
            torch.Tensor: The proposal distribution.
            a value : (for 1D) 
            # x[idx^1] should refere to the conditined one
            # ^ bitwise operator, same 0, else 1
            # @ to get the square of L2 norm
        """
        return (-(.25 / self.lr_fn(self.counter)) *
                (self.x[idx] - self.x[idx ^ 1] -
                 self.lr_fn(self.counter) * self.grad[idx ^ 1])).squeeze() @ (self.x[idx] - self.x[idx ^ 1] - self.lr_fn(self.counter) * self.grad[idx ^ 1]).squeeze()

    def sample_prob(self) -> torch.Tensor:
        """
        Calculate the acceptance probability for Metropolis-Hastings.

        Returns:
            torch.Tensor: The acceptance probability.
            
        # loss is the negative logp
        # The dimension should be 1. 
        # This is not pratical for high-dimensional data
        """
        return torch.exp(-self.loss[1] + self.loss[0]) * \
            torch.exp(self.proposal_dist(0) - self.proposal_dist(1))

    def decay_fn(self,
                 lr: float = 1e-2,
                 lr_final: float = 1e-4,
                 max_itr: float = 1e4) -> callable:
        """
        Generate a learning rate decay function.

        Args:
            lr (float, optional): Initial learning rate. Default is 1e-2.
            lr_final (float, optional): Final learning rate. Default is 1e-4.
            max_itr (float, optional): Maximum number of iterations. Default is
                1e4.

        Returns:
            callable: Learning rate decay function.
        """
        gamma = -0.55
        b = max_itr / ((lr_final / lr)**(1 / gamma) - 1.0)
        a = lr / (b**gamma)

        def lr_fn(t: float,
                  a: float = a,
                  b: float = b,
                  gamma: float = gamma) -> float:
            return a * ((b + t)**gamma)

        return lr_fn

    def lr_decay(self):
        """
        Decay the learning rate of the optimizer.
        """
        for param_group in self.optim.param_groups:
            param_group['lr'] = self.lr_fn(self.counter)
