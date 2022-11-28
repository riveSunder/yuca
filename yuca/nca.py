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

from yuca.multiverse import CA

class UNCA(CA):
    """
    UNCA - Universal Neural Cellular Automata

    Similar to and inheriting from CA

    uses neural networks for genesis and peristence or growth functions
    """

    def __init__(self, **kwargs):
        super(UNCA, self).__init__()

        # CA mode. Options are 'neural' or 'functional'.
        # these are initialized in parent class CA but won't be used by UNCA
#        self.ca_mode = query_kwargs("ca_mode", "functional", **kwargs)
#        self.genesis_fns = []
#        self.persistence_fns = []


    def load_config(self, config):
        
        # TODO: Implement for nca
        pass
        
    def make_config(self):

        # TODO: Implement for nca
        pass
        #return copy.deepcopy(config)

    def save_config(self, filepath, config=None):

        # TODO: Implement for nca
        pass

    def default_init3(self):

        # TODO: consolidate initializations
        pass

    def default_init2(self):
        # TODO: consolidate initializations
        pass

    def default_init(self):

        # TODO: consolidate initializations
        pass

    def random_init(self):
        
        # TODO: consolidate initializations
        pass

        
    def initialize_weight_layer(self):
        
        self.weights_layer = nn.Sequential(\
                nn.Conv2d(\
                    self.internal_channels, self.internal_channels, 1, \
                    padding=0, \
                    padding_mode = self.conv_mode, bias=False),\
                nn.ReLU(), \
                nn.Conv2d(\
                    self.internal_channels, self.external_channels, 1, \
                    padding=0, \
                    padding_mode = self.conv_mode, bias=False)\
                )

    def add_genesis_fn(self,  config):
        # Not used in nca
        pass

    def add_persistence_fn(self, config):
        # Not used in nca
        pass
    
    def include_parameters(self):
        # Not used in nca?
        # (neural parameters should be registered/tracked automatically)
        pass

    def persistence(self, neighborhoods):
        # Not used in nca
        pass
    
    def genesis(self, neighborhoods):
        # Not used in nca
        pass

    def id_conv(self, universe):
        # shared with CCA
        
        return self.id_layer(universe)
    
    def neighborhood_conv(self, universe):
        # shared with CCA

        return self.neighborhood_layer(universe)

    def alive_mask(self, universe):
        """
        zero out cells not meeting a threshold in the alpha channel
        """

        # shared with CCA

        alive_mask = torch.zeros_like(universe[:, 3:4, :, :])

        alive_mask[universe[:, 3:4, :, :] > self.alive_threshold] = 1.0

        return universe * alive_mask

    def update_universe(self, identity, neighborhoods):

        # TODO: NCA version
        pass

        #return update 

    def forward(self, universe, mode=0):

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
        #new_universe = self.weights_layer(new_universe)
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

    def fit_growth(self, target, **kwargs):
        
        # remove entirely 

    def to_device(self, my_device):
        """
        additional functionality beyong `to` function from nn.Module
        to ensure all parameters get moved for CCA
        """
        # TODO: implement for nca (should be simpler) 
        pass
        
        self.to(my_device)
        self.my_device = my_device
        self.id_layer.to(my_device)
        self.neighborhood_layer.to(my_device)
        self.weights_layer.to(my_device)

    def get_genesis_params(self):
        # Not used in nca
        pass

    def get_persistence_params(self):
        # Not used in nca
        pass

    def get_params(self):
        # TODO: implement nca version (simpler)
    
        params = np.array([])

        for hh, param in enumerate(self.weights_layer.named_parameters()):
            params = np.append(params, param[1].detach().numpy().ravel())

        return params

    def set_params(self, params):
        # TODO: implement nca version (simpler)

        param_start = 0

        # weights are not a learnable parameter for functional CA
        for hh, param in self.weights_layer.named_parameters():
            if not len(param.shape):
                param = param.reshape(1)

                param_stop = param_start + reduce(lambda x,y: x*y, param.shape)
                param[:] = nn.Parameter( \
                        torch.tensor( \
                        params[param_start:param_stop].reshape(param.shape),\
                        requires_grad = self.use_grad), \
                        requires_grad = self.use_grad)

                param_start = param_stop

    def no_grad(self):

        self.use_grad = False

        for hh, param in enumerate(self.weights_layer.parameters()):
            param.requires_grad = False
        

if __name__ == "__main__":
    pass
