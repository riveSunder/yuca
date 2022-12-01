import os
import copy

import numpy as np
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from yuca.utils import save_fig_sequence, \
        make_gif, \
        query_kwargs

from yuca.kernels import get_kernel, \
        get_laplacian_kernel


from yuca.ca.common import CA

class RxnDfn(CA):

    def __init__(self):
        super().__init__()
        
        self.internal_channels = 2
        self.external_channels = 2

        self.default_init()
        self.reset()

    def default_init(self):

        self.add_neighborhood_kernel()
        self.initialize_neighborhood_layer()

        self.add_identity_kernel()
        self.initialize_id_layer()

    
        """
        Gray-Scott model 
        updates accordin gto 

        $\frac{\partial u}{\partial t} = diffusion_u \nabla^2 u - uv^2 + f(1-u)$

        $\frac{\partial v}{\partial t} = diffusion_v \nabla^2 v + uv^2 - (f + k)v$

        """

        # Gray-Scott parameters
        # nominal U-Skate world
#        self.f = torch.tensor([0.062])
#        self.k = torch.tensor([0.06093])
#        self.diffusion_u = torch.tensor([0.64])
#        self.diffusion_v = torch.tensor([0.32])
#        self.dt = torch.tensor([0.2])
        self.f = torch.tensor([0.062])
        self.k = torch.tensor([0.06093])
        self.diffusion_u = torch.tensor([0.64])
        self.diffusion_v = torch.tensor([0.32])
        # time step
        self.dt = torch.tensor([0.2])
        # spatial step 
        self.dx = torch.tensor([1.0]) #1/143.
        # 1/143. is from mrob


    def add_neighborhood_kernel(self, kernel=None):
        """
        add Laplacian kernel (kernel arg is ignored).
        """

        kernel = get_laplacian_kernel()

        kernel_dims = len(kernel.shape)
        
        error_msg = f"neighborhood kernel expected to have dim 4," \
                f" had dim {kernel_dims} instead"

        assert len(kernel.shape) == 4, error_msg

        dim_x = kernel.shape[-1]
        dim_y = kernel.shape[-2]

        error_msg = f"expected square kernel, got {dim_x} by {dim_y}"

        assert dim_x == dim_y, error_msg

        kernel = torch.cat([kernel, kernel], dim=0)
        self.neighborhood_kernels = kernel 
        self.neighborhood_dim = dim_x 

    def initialize_neighborhood_layer(self):
        """
        initialized a convolutional layer containing neighborhod functions
        """

        padding = (self.neighborhood_dim - 1) // 2 

        groups = 1 if self.internal_channels % self.external_channels \
                else self.external_channels

        self.neighborhood_layer = nn.Conv2d(self.external_channels, \
                self.internal_channels, \
                self.neighborhood_dim, padding=padding, \
                groups=groups, \
                padding_mode = self.conv_mode, bias=False)


        for param in self.neighborhood_layer.named_parameters():
            param[1].requires_grad = False
            param[1][:] = self.neighborhood_kernels

    def update_universe(self, identity, neighborhoods):
        """
        Gray-Scott model 
        updates accordin gto 

        $\frac{\partial u}{\partial t} = diffusion_u \nabla^2 u - uv^2 + f(1-u)$

        $\frac{\partial v}{\partial t} = diffusion_v \nabla^2 v + uv^2 - (f + k)v$

        args:
            identity - current state
            neighborhoods - laplacian of current state

        both identity and neighborhoods have dims of NxCxHxW, where C (channels) is 2,
        representing the values for species u and species v
        """

        update = 0 * identity

        nabla_u = self.dx * neighborhoods[:,0,:,:] 
        nabla_v = self.dx * neighborhoods[:,1,:,:] 
        u = identity[:,0,:,:]
        v = identity[:,1,:,:]

        # species u
        update[:,0,:,:] = self.diffusion_u * nabla_u - u*v*v + self.f*(1-u)
        # species v
        update[:,1,:,:] = self.diffusion_v * nabla_v + u*v*v - v*(self.f+self.k)

        return update

    
    def alive_mask(self, universe):
        # no alive_mask in Gray-Scott reaction-diffusion models
        return universe

    def forward(self, universe):

        if universe.shape[1] >= 4:
            universe = self.alive_mask(universe)

        identity = self.id_conv(universe)
        neighborhoods = self.neighborhood_conv(universe)

        update = self.update_universe(identity, neighborhoods)
        
        new_universe = universe + self.dt * update #, 0, 1.0)

        self.t_count += self.dt

        return new_universe


    def get_params(self):

        params = np.array([])

        for param in [self.diffusion_u, self.diffusion_v, self.f, self.k]:
            params = np.append(params, param.detach().numpy().ravel())

        return params

    def set_params(self, params):

        param_start = 0

        
        param_stop = param_start + 1
        self.diffusion_u = torch.tensor(params[param_start], requires_grad = self.use_grad)
        param_start += 1
        self.diffusion_v = torch.tensor(params[param_start], requires_grad = self.use_grad)
        param_start += 1
        self.f = torch.tensor(params[param_start], requires_grad = self.use_grad)
        param_start += 1
        self.k = torch.tensor(params[param_start], requires_grad = self.use_grad)


    def include_parameters(self):

        for jj, param in zip(["diffusion_u", "diffusion_v", "f", "k"],\
                [self.diffusion_u, self.diffusion_v, self.f, self.k]):
            self.register_parameter(f"rd_{jj}", param)
