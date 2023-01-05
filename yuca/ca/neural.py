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
         
from yuca.kernels import get_kernel, \
        get_gaussian_kernel, \
        get_dogaussian_kernel, \
        get_gaussian_edge_kernel, \
        get_cosx2_kernel, \
        get_dogaussian_edge_kernel
         
from yuca.patterns import get_orbium, \
        get_smooth_puffer


import matplotlib.pyplot as plt

from yuca.ca.common import CA

class NCA(CA):
    """
    NCA - Neural Cellular Automata

    Similar to and inheriting from CA

    uses neural networks for genesis and peristence or growth functions
    """

    def __init__(self, **kwargs):
        self.hidden_channels = query_kwargs("hidden_channels", 128, **kwargs)
        super(NCA, self).__init__(**kwargs)

        # CA mode. Options are 'neural' or 'functional'.
        # these are initialized in parent class CA but won't be used by UNCA
#        self.ca_mode = query_kwargs("ca_mode", "functional", **kwargs)
#        self.genesis_fns = []
#        self.persistence_fns = []


        self.default_init()
        self.reset()

    def default_init(self):

        self.add_identity_kernel()
        self.initialize_id_layer()

        nbhd_kernels = None 

        for mm in range(self.internal_channels):

            my_radius = self.kernel_radius

            mu = np.random.rand() 
            sigma = np.random.rand() 

            mu = np.clip(mu, 0.05, 0.95)
            sigma = np.clip(sigma, 0.0005, 0.1)
            
            nbhd_kernel = get_gaussian_kernel(radius=my_radius, \
                    mu=mu, sigma=sigma)

            if nbhd_kernels is None:
                nbhd_kernels = nbhd_kernel
            else:
                nbhd_kernels = torch.cat([nbhd_kernels, nbhd_kernel], dim=0)

        self.add_neighborhood_kernel(nbhd_kernels)
        self.initialize_neighborhood_layer()

        self.initialize_weight_layer()

        self.dt = 0.1

    def load_config(self, config):

        self.config = config

        if "identity_kernel_config" not in config.keys():
            self.add_identity_kernel()
        else: 
            id_kernel = get_kernel(config["identity_kernel_config"])
            self.add_identity_kernel(kernel = id_kernel)

        self.initialize_id_layer()

        nbhd_kernel = get_kernel(config["neighborhood_kernel_config"])
        self.neighborhood_kernel_config = config["neighborhood_kernel_config"]

        self.add_neighborhood_kernel(nbhd_kernel)
        self.initialize_neighborhood_layer()

        self.set_params(config["params"])

        if "dt" in config.keys():
            self.dt = config["dt"]
        else: 
            self.dt = 0.1

        self.initialize_weight_layer()
        self.include_parameters()

    def restore_config(self, filepath):
        if "\n" in filepath:
            filepath = filepath.replace("\n","")

        file_directory = os.path.abspath(__file__).split("/")
        root_directory = os.path.join(*file_directory[:-3])
        default_directory = os.path.join("/", root_directory, "ca_configs")

        if os.path.exists(filepath):

            config = np.load(filepath, allow_pickle=True).reshape(1)[0]
            self.load_config(config)
            print(f"config restored from {filepath}")

        elif os.path.exists(os.path.join(default_directory, filepath)):
            
            filepath = os.path.join(default_directory, filepath)
            config = np.load(filepath, allow_pickle=True).reshape(1)[0]
            self.load_config(config)
            print(f"config restored from {filepath}")
        
        else:

            print(f"default directory: {default_directory}")
            print(f"attempted to read {filepath}, not found")
            #assert False, f"{filepath} not found"
        
    def make_config(self):

        config = {}

        # config for identity kernel
        if self.id_kernel_config is None:
            
            id_kernel_config = {}
            id_kernel_config["name"] = "InnerMoore"

        else:
            id_kernel_config = self.id_kernel_config

        # config for neighborhood kernel(s)
        if self.neighborhood_kernel_config is None:
            print("kernel config is missing, assuming GaussianMixture")
            #assert False,  "not implemented exception"
            neighborhood_kernel_config = "GaussianMixture"
            neighborhood_kernel_config = {}
            neighborhood_kernel_config["name"] = "GaussianMixture"
            neighborhood_kernel_config["kernel_kwargs"] = {}
            neighborhood_kernel_config["radius"] = self.kernel_radius


        else:
            neighborhood_kernel_config = self.neighborhood_kernel_config

        # nca params
        config["params"] = self.get_params()
            
        config["id_kernel_config"] = id_kernel_config
        config["neighborhood_kernel_config"] = neighborhood_kernel_config

        return copy.deepcopy(config)

    def save_config(self, filepath, config=None):

        if config is None:
            config = self.make_config()

        np.save(filepath, config)

    def initialize_weight_layer(self):
        
        self.weights_layer = nn.Sequential(\
                nn.Conv2d(\
                    self.external_channels + self.internal_channels,\
                    self.hidden_channels, 1, \
                    padding=0, \
                    padding_mode = self.conv_mode, bias=True),\
                nn.ReLU(), \
                nn.Conv2d(\
                    self.hidden_channels, self.external_channels, 1, \
                    padding=0, \
                    padding_mode = self.conv_mode, bias=False)\
                )
    

    def id_conv(self, universe):
        # shared with CCA
        
        return self.id_layer(universe)
    
    def neighborhood_conv(self, universe):
        # shared with CCA

        return self.neighborhood_layer(universe)

    def update_universe(self, identity, neighborhoods):

        # TODO: NCA version
        model_input = torch.cat([identity, neighborhoods], dim=1)
        update = self.weights_layer(model_input)
        return update 

    def to_device(self, my_device):
        """
        additional functionality beyong `to` function from nn.Module
        to ensure all parameters get moved for CCA
        """
        # TODO: implement for nca (should be simpler) 
        
        self.no_grad()
        self.to(my_device)
        self.my_device = my_device
        self.id_layer.to(my_device)
        self.neighborhood_layer.to(my_device)
        self.weights_layer.to(my_device)


    def get_params(self):
        # TODO: implement nca version (simpler)
    
        params = np.array([])


        for hh, param in enumerate(self.weights_layer.named_parameters()):
            params = np.append(params, param[1].detach().cpu().numpy().ravel())

        return params

    def set_params(self, params):

        self.no_grad()
        # TODO: implement nca version (simpler)

        param_start = 0

        for hh, param in self.weights_layer.named_parameters():

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

    def include_parameters(self):
        pass
        

if __name__ == "__main__":
    pass
