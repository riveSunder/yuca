import os
import copy

import numpy as np
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from yuca.activations import Gaussian, \
        DoubleGaussian, \
        GaussianMixture, \
        DoGaussian, \
        Polynomial, \
        SmoothIntervals, \
        Identity

import yuca.utils as utils
from yuca.utils import save_fig_sequence, \
        make_gif, \
        query_kwargs
         
from yuca.configs import get_orbium_config, \
        get_geminium_config

from yuca.kernels import get_kernel, \
        get_gaussian_kernel, \
        get_dogaussian_kernel, \
        get_gaussian_edge_kernel, \
        get_cosx2_kernel, \
        get_dogaussian_edge_kernel
         
from yuca.patterns import get_orbium, \
        get_smooth_puffer

import matplotlib.pyplot as plt

class CA(nn.Module):
    """
    """

    def __init__(self, **kwargs):
        super(CA, self).__init__()

        self.kernel_radius = query_kwargs("kernel_radius", 13, **kwargs)
        self.my_device = query_kwargs("device", "cpu", **kwargs)
        
        self.conv_mode = query_kwargs("conv_mode", "circular", **kwargs)

        # CA mode. Options are 'neural' or 'functional'.
        self.ca_mode = query_kwargs("ca_mode", "functional", **kwargs)


        self.external_channels = query_kwargs("external_channels", 1, **kwargs)
        self.internal_channels = query_kwargs("internal_channels", 1, **kwargs)
        self.alive_threshold = query_kwargs("alive_threshold", 0.1, **kwargs)

        self.use_grad = query_kwargs("use_grad", False, **kwargs)
        self.tag = query_kwargs("tag", "notag", **kwargs)


        self.neighborhood_kernels = None 
        self.neighborhood_dim = None

        self.genesis_fns = []
        self.persistence_fns = []

        self.id_kernel_config = None
        self.neighborhood_kernel_config = None
        self.genesis_fn_config = None
        self.persistence_fn_config = None

        self.input_filepath = query_kwargs("input_filepath", None, **kwargs)
            

    def reset(self):
        
        self.t_count = 0.0
        
    def add_identity_kernel(self, **kwargs):
        
        if "kernel" in kwargs.keys():
            self.id_kernel = kwargs["kernel"]

            if len(self.id_kernel.shape) == 2:
                self.id_kernel = torch.reshape(self.id_kernel, \
                        (1, 1, \
                        self.id_kernel.shape[0], \
                        self.id_kernel.shape[1]))
            elif len(self.id_kernel.shape) == 3:
                self.id_kernel = torch.reshape(self.id_kernel, \
                        (1, self.id_kernel.shape[0], \
                        self.id_kernel.shape[1]))

        else:
            self.id_kernel = torch.tensor([[[\
                    [0, 0.0, 0,], \
                    [0, 1.0, 0], \
                    [0, 0.0, 0]]]])

        kernel_dims = len(self.id_kernel.shape)
        
        error_msg = f"id kernel expected to have dim 4," \
                f" had dim {kernel_dims} instead"

        assert len(self.id_kernel.shape) == 4, error_msg

        dim_x = self.id_kernel.shape[-1]
        dim_y = self.id_kernel.shape[-2]

        error_msg = f"expected square kernel, got {dim_x} by {dim_y}"

        assert dim_x == dim_y, error_msg

        self.id_dim = dim_x


    def initialize_id_layer(self):

        padding = (self.id_dim - 1) // 2 

        groups = 1 if self.internal_channels % self.external_channels \
                else self.external_channels

        if groups != self.internal_channels or groups != self.external_channels:
            print(f"warning, id_layer has {groups} groups, but "\
                    f"{self.internal_channels},{self.external_channels} channels, "\
                    f"id convolution will mix channels!")

        self.id_layer = nn.Conv2d(self.external_channels, \
                self.internal_channels, self.id_dim, padding=padding, \
                groups = groups,\
                padding_mode = self.conv_mode, bias=False)

        for param in self.id_layer.named_parameters():
            param[1].requires_grad = False
            param[1][:] = self.id_kernel
        

    def add_neighborhood_kernel(self, kernel):
        """
        add kernels defining CA neighborhoods.

        Planned functionality will make it possible to
        add additional neighborhood kernels to an existing
        stack, but for now the input argument kernel replaces
        any existing neighborhood kernels
        """

        if type(kernel) is not torch.Tensor:
            kernel = torch.tensor(kernel, requires_grad=False)
        else:
            kernel = kernel.clone().detach().requires_grad_(False)

        if len(kernel.shape) == 2:
            kernel = torch.reshape(kernel, \
                    (1, 1, \
                    kernel.shape[0], \
                    kernel.shape[1]))
        elif len(kernel.shape) == 3:
            kernel = torch.reshape(kernel, \
                    (1, kernel.shape[0], \
                    kernel.shape[1]))

        kernel_dims = len(kernel.shape)
        
        error_msg = f"id kernel expected to have dim 4," \
                f" had dim {kernel_dims} instead"

        assert len(kernel.shape) == 4, error_msg

        dim_x = kernel.shape[-1]
        dim_y = kernel.shape[-2]

        error_msg = f"expected square kernel, got {dim_x} by {dim_y}"

        assert dim_x == dim_y, error_msg

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
        
    def id_conv(self, universe):
        """
        """
        
        return self.id_layer(universe)
    
    def neighborhood_conv(self, universe):

        return self.neighborhood_layer(universe)

    def update_universe(self, identity, neighborhoods):

        return identity + neighborhoods

    def alive_mask(self, universe):
        """
        zero out cells not meeting a threshold in the alpha channel
        
        """

        alive_mask = torch.zeros_like(universe[:, 3:4, :, :])

        alive_mask[universe[:, 3:4, :, :] > self.alive_threshold] = 1.0

        return universe * alive_mask


    def forward(self, universe):

        if universe.shape[1] >= 4:
            universe = self.alive_mask(universe)

        if universe.dtype == torch.float16 and torch.device(self.my_device).type != "cuda":
            identity = self.id_conv(universe.to(torch.float32)).to(torch.float16)
            neighborhoods = self.neighborhood_conv(universe.to(torch.float32)).to(torch.float16)

            update = self.update_universe(identity, neighborhoods)
            
            new_universe = torch.clamp(universe.to(torch.float32) + self.dt * update.to(torch.float32), 0, 1.0).to(torch.float16)
        else:
 
            identity = self.id_conv(universe)
            neighborhoods = self.neighborhood_conv(universe)

            update = self.update_universe(identity, neighborhoods)
            
            new_universe = torch.clamp(universe + self.dt * update, 0, 1.0)

        self.t_count += self.dt

        return new_universe

    def get_frame(self, universe, mode=0):

        identity = self.id_conv(universe)
        neighborhoods = self.neighborhood_conv(universe)

        update = self.update_universe(identity, neighborhoods)

        new_universe = torch.clamp(universe + self.dt * update, 0, 1.0)

        if mode == 0:
            return new_universe
        elif mode == 1:
            return new_universe, neighborhoods
        elif mode == 2:
            return new_universe, update

        elif mode == 3:
            return new_universe, neighborhoods, update 
        elif mode == 4:
            return new_universe, universe, neighborhoods, update 

        

if __name__ == "__main__":

    pass
