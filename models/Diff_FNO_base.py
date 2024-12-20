from functools import partialmethod

import torch.nn as nn
import torch.nn.functional as F

import neuralop

from neuralop.layers.embeddings import GridEmbeddingND, GridEmbedding2D
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.padding import DomainPadding
from neuralop.layers.local_fno_block import LocalFNOBlocks
from neuralop.layers.mlp import MLP
from neuralop.models.base_model import BaseModel

class Diff_FNO(BaseModel, name='Diff_FNO'):
    """N-Dimensional Fourier Neural Operator

    Parameters
    ----------
    n_modes : int tuple
        number of modes to keep in Fourier Layer, along each dimension
        The dimensionality of the FNO is inferred from ``len(n_modes)``
    hidden_channels : int
        width of the FNO (i.e. number of channels)
    in_channels : int, optional
        Number of input channels, by default 3
    out_channels : int, optional
        Number of output channels, by default 1
    lifting_channels : int, optional
        number of hidden channels of the lifting block of the FNO, by default 256
    projection_channels : int, optional
        number of hidden channels of the projection block of the FNO, by default 256
    n_layers : int, optional
        Number of Fourier Layers, by default 4
    positional_embedding : str literal | GridEmbedding2D | GridEmbeddingND | None
        if "grid", appends a grid positional embedding with default settings to 
        the last channels of raw input. Assumes the inputs are discretized
        over a grid with entry [0,0,...] at the origin and side lengths of 1.
        If an initialized GridEmbedding, uses this module directly
        See `neuralop.embeddings.GridEmbeddingND` for details
        if None, does nothing
    max_n_modes : None or int tuple, default is None
        * If not None, this allows to incrementally increase the number of
          modes in Fourier domain during training. Has to verify n <= N
          for (n, m) in zip(max_n_modes, n_modes).

        * If None, all the n_modes are used.

        This can be updated dynamically during training.
    fno_block_precision : str {'full', 'half', 'mixed'}
        if 'full', the FNO Block runs in full precision
        if 'half', the FFT, contraction, and inverse FFT run in half precision
        if 'mixed', the contraction and inverse FFT run in half precision
    stabilizer : str {'tanh'} or None, optional
        By default None, otherwise tanh is used before FFT in the FNO block
    use_mlp : bool, optional
        Whether to use an MLP layer after each FNO block, by default False
    mlp_dropout : float , optional
        droupout parameter of MLP layer, by default 0
    mlp_expansion : float, optional
        expansion parameter of MLP layer, by default 0.5
    non_linearity : nn.Module, optional
        Non-Linearity module to use, by default F.gelu
    norm : Literal["ada_in", "group_norm", "instance_norm"], optional
        Normalization layer to use, by default None
    preactivation : bool, default is False
        if True, use resnet-style preactivation
    fno_skip : {'linear', 'identity', 'soft-gating'}, optional
        Type of skip connection to use in fno, by default 'linear'
    mlp_skip : {'linear', 'identity', 'soft-gating'}, optional
        Type of skip connection to use in mlp, by default 'soft-gating'
    separable : bool, default is False
        if True, use a depthwise separable spectral convolution
    factorization : str or None, {'tucker', 'cp', 'tt'}
        Tensor factorization of the parameters weight to use, by default None.
        * If None, a dense tensor parametrizes the Spectral convolutions
        * Otherwise, the specified tensor factorization is used.
    joint_factorization : bool, optional
        Whether all the Fourier Layers should be parametrized by a single tensor
        (vs one per layer), by default False
    rank : float or rank, optional
        Rank of the tensor factorization of the Fourier weights, by default 1.0
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False
    implementation : {'factorized', 'reconstructed'}, optional, default is 'factorized'
        If factorization is not None, forward mode to use::
        * `reconstructed` : the full weight tensor is reconstructed from the
          factorization and used for the forward pass
        * `factorized` : the input is directly contracted with the factors of
          the decomposition
    decomposition_kwargs : dict, optional, default is {}
        Optionaly additional parameters to pass to the tensor decomposition
    domain_padding : None, float, or List[float], optional
        If not None, percentage of padding to use, by default None
        To vary the percentage of padding used along each input dimension,
        pass in a list of percentages e.g. [p1, p2, ..., pN] such that
        p1 corresponds to the percentage of padding along dim 1, etc.
    domain_padding_mode : {'symmetric', 'one-sided'}, optional
        How to perform domain padding, by default 'one-sided'
    fft_norm : str, optional
        by default 'forward'
        
    FiniteDifferenceConvolution Params
    ----------------------------------
    diff_layers : bool list, optional
        Must be same length as n_layers, dictates whether to include a
        differential kernel parallel connection at each layer
    fin_diff_implementation : str in ['subtract_middle', 'subtract_all'], optional
        Implementation type for FiniteDifferenceConvolution.
        See differential_conv.py.
    conv_padding_mode : str in ['periodic', 'circular', 'replicate', 'reflect', 'zeros'], optional
        Padding mode for spatial convolution kernels.
    default_grid_res : int or None, optional
        Proportional to default input shape of last spatial dimension. If 
        None, inferred from data. This is used for defining the appropriate
        scaling of the differential kernel.
    fin_diff_kernel_size : odd int, optional
        Conv kernel size for finite difference convolution.
    mix_derivatives : bool, optional
        Whether to mix derivatives across channels
    """

    def __init__(
        self,
        n_modes,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        positional_embedding="grid",
        output_scaling_factor=None,
        max_n_modes=None,
        fno_block_precision="full",
        use_mlp=False,
        mlp_dropout=0,
        mlp_expansion=0.5,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        preactivation=False,
        fno_skip="linear",
        mlp_skip="soft-gating",
        separable=False,
        factorization=None,
        rank=1.0,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="one-sided",
        fft_norm="forward",
        SpectralConv=SpectralConv,
        diff_layers=[True],   ## diff layer
        fin_diff_implementation='subtract_middle',
        conv_padding_mode='periodic',
        default_grid_res=None,
        fin_diff_kernel_size=3,
        mix_derivatives=True,
        **kwargs
    ):
        super().__init__()
        self.n_dim = len(n_modes)

        # See the class' property for underlying mechanism
        # When updated, change should be reflected in fno blocks
        self._n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = (fno_skip,)
        self.mlp_skip = (mlp_skip,)
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.fno_block_precision = fno_block_precision
        
        # diff
        self.diff_layers = diff_layers * self.n_layers

        if positional_embedding == "grid":
            spatial_grid_boundaries = [[0., 1.]] * self.n_dim
            self.positional_embedding = GridEmbeddingND(dim=self.n_dim, grid_boundaries=spatial_grid_boundaries)
        elif isinstance(positional_embedding, GridEmbedding2D):
            if self.n_dim == 2:
                self.positional_embedding = positional_embedding
            else:
                raise ValueError(f'Error: expected {self.n_dim}-d positional embeddings, got {positional_embedding}')
        elif isinstance(positional_embedding, GridEmbeddingND):
            self.positional_embedding = positional_embedding
        elif positional_embedding == None:
            self.positional_embedding = None
        else:
            raise ValueError(f"Error: tried to instantiate FNO positional embedding with {positional_embedding},\
                              expected one of \'grid\', GridEmbeddingND")
        
        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode,
                output_scaling_factor=output_scaling_factor,
            )
        else:
            self.domain_padding = None

        self.domain_padding_mode = domain_padding_mode

        if output_scaling_factor is not None and not joint_factorization:
            if isinstance(output_scaling_factor, (float, int)):
                output_scaling_factor = [output_scaling_factor] * self.n_layers
        self.output_scaling_factor = output_scaling_factor

        self.fno_blocks = LocalFNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            output_scaling_factor=output_scaling_factor,
            use_mlp=use_mlp,
            mlp_dropout=mlp_dropout,
            mlp_expansion=mlp_expansion,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            norm=norm,
            preactivation=preactivation,
            fno_skip=fno_skip,
            mlp_skip=mlp_skip,
            max_n_modes=max_n_modes,
            fno_block_precision=fno_block_precision,
            rank=rank,
            fft_norm=fft_norm,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
            SpectralConv=SpectralConv,
            n_layers=n_layers,
            diff_layers=self.diff_layers, # diff layers
            fin_diff_implementation=fin_diff_implementation,
            conv_padding_mode=conv_padding_mode,
            default_grid_res=default_grid_res,
            fin_diff_kernel_size=fin_diff_kernel_size,
            mix_derivatives=mix_derivatives,
            **kwargs
        )

        lifting_in_channels = self.in_channels
        if self.positional_embedding is not None:
            lifting_in_channels += self.n_dim
        # if lifting_channels is passed, make lifting an MLP
        # with a hidden layer of size lifting_channels
        if self.lifting_channels:
            self.lifting = MLP(
                in_channels=lifting_in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim,
            )
        # otherwise, make it a linear layer
        else:
            self.lifting = MLP(
                in_channels=lifting_in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.hidden_channels,
                n_layers=1,
                n_dim=self.n_dim,
            )
        self.projection = MLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )

    def forward(self, x, output_shape=None, **kwargs):
        """TFNO's forward pass

        Parameters
        ----------
        x : tensor
            input tensor
        output_shape : {tuple, tuple list, None}, default is None
            Gives the option of specifying the exact output shape for odd shaped inputs.
            * If None, don't specify an output shape
            * If tuple, specifies the output-shape of the **last** FNO Block
            * If tuple list, specifies the exact output-shape of each FNO Block
        """

        if output_shape is None:
            output_shape = [None]*self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None]*(self.n_layers - 1) + [output_shape]

        # append spatial pos embedding if set
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)
        
        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes
