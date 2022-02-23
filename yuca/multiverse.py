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

        self.internal_channels = query_kwargs("internal_channels", 1, **kwargs)
        self.external_channels = query_kwargs("external_channels", 1, **kwargs)
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

        if "orbi" in self.tag.lower():
            self.load_config(get_orbium_config(radius=self.kernel_radius))
        elif "len" in self.tag.lower():
            self.load_config(get_orbium_config(radius=self.kernel_radius))
        elif "gemin" in self.tag.lower():
            self.load_config(get_geminium_config(radius=self.kernel_radius))
        elif "init3" in self.tag.lower():
            self.default_init3()
        else:
            self.random_init()

        self.input_filepath = query_kwargs("input_filepath", None, **kwargs)

        if self.input_filepath is not None:
            my_data = np.load(self.input_filepath, allow_pickle=True).reshape(1)[0]

            my_params = my_data["elite_params"][-1][0]

            self.set_params(my_params)

        self.reset()

    def reset(self):
        
        self.t_count = 0.0

    def load_config(self, config):

        self.genesis_fns = []
        self.persistence_fns = []

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

        self.add_genesis_fn(config["genesis_config"])
        self.add_persistence_fn(config["persistence_config"])

        self.genesis_fn_config = config["genesis_config"]
        self.persistence_fn_config = config["persistence_config"]

        if "dt" in config.keys():
            self.dt = config["dt"]
        else: 

            self.dt = 0.1

        self.initialize_weight_layer()
        self.include_parameters()

    def restore_config(self, filepath):

        default_directory = os.path.split(\
                os.path.split(os.path.realpath(__file__))[0])[0]
        default_directory = os.path.join(default_directory, "ca_configs")

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

            print(f"attempted to read {filepath}, not found")
            assert False, f"{filepath} not found"
        
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
            print("kernel config is missing")
            assert False,  "not implemented exception"
        else:
            neighborhood_kernel_config = self.neighborhood_kernel_config

        # config for genesis and persistence functions

        # genesis 
        if self.genesis_fn_config is None:
            print("genesis fn config is missing")
            assert False,  "not implemented exception"
        else:
            genesis_config = self.genesis_fn_config

        #if "parameters" in genesis_config.keys():

        #    genesis_config["parameters"] = self.get_genesis_params()
            
        # persistence
        if self.persistence_fn_config is None:
            print("genesis fn config is missing")
            assert False,  "not implemented exception"
        else:
            persistence_config = self.persistence_fn_config

        #if "parameters" in persistence_config.keys():

        #    persistence_config["parameters"] = self.get_persistence_params()
            
        config["id_kernel_config"] = id_kernel_config
        config["neighborhood_kernel_config"] = neighborhood_kernel_config
        config["genesis_config"] = genesis_config
        config["persistence_config"] = persistence_config

        return copy.deepcopy(config)

    def save_config(self, filepath, config=None):

        if config is None:
            config = self.make_config()

        np.save(filepath, config)

    def default_init3(self):

        self.genesis_fns = []
        self.persistence_fns = []

        self.add_identity_kernel()
        self.initialize_id_layer()

        nbhd_kernel = get_cosx2_kernel(radius=self.kernel_radius)

        self.add_neighborhood_kernel(nbhd_kernel)
        self.initialize_neighborhood_layer()

        self.initialize_weight_layer()

        self.dt = 0.1

        params_g = torch.tensor([0.22, 0.008, 0.525, 0.007])
        params_p = torch.tensor([0.15, 0.008, .295, 0.007])

        genesis_config = {"name": "GaussianMixture",\
                "parameters": params_g, \
                "mode": 1}

        persistence_config = {"name": "GaussianMixture",\
                "parameters": params_p, \
                "mode": 1}

        self.add_genesis_fn(genesis_config)
        self.add_persistence_fn(persistence_config)

        self.include_parameters()

    def default_init2(self):

        self.genesis_fns = []
        self.persistence_fns = []
        
        self.add_identity_kernel()
        self.initialize_id_layer()

        nbhd_kernel = get_gaussian_kernel(radius=self.kernel_radius)

        self.add_neighborhood_kernel(nbhd_kernel)
        self.initialize_neighborhood_layer()

        self.initialize_weight_layer()

        self.dt = 0.1
        

        params_g = torch.tensor([0.15, 0.005, 0.525, 0.005])
        params_p = torch.tensor([0.22, 0.005, .295, 0.005])

        genesis_config = {"name": "GaussianMixture",\
                "parameters": params_g, \
                "mode": 1}

        persistence_config = {"name": "GaussianMixture",\
                "parameters": params_p, \
                "mode": 1}

        self.add_genesis_fn(genesis_config)
        self.add_persistence_fn(persistence_config)

        self.include_parameters()

    def default_init(self):

        self.genesis_fns = []
        self.persistence_fns = []
        
        self.add_identity_kernel()
        self.initialize_id_layer()

        nbhd_kernel = get_gaussian_kernel(radius=self.kernel_radius)

        self.add_neighborhood_kernel(nbhd_kernel)
        self.initialize_neighborhood_layer()

        self.initialize_weight_layer()

        self.dt = 0.1
        
        genesis_config = {"name": "Gaussian",\
                "mu": 0.15, \
                "sigma": 0.015, \
                "mode": 1}

        self.add_genesis_fn(genesis_config)
        self.add_persistence_fn(genesis_config)

        self.include_parameters()

    def random_init(self):

        self.genesis_fns = []
        self.persistence_fns = []
        self.add_identity_kernel()
        self.initialize_id_layer()

        nbhd_kernels = None 

        for mm in range(self.internal_channels):

                
            my_radius = 1
            if mm == 0:
                nbhd_kernel = torch.zeros(1,1, my_radius*2+1, my_radius*2+1)
                nbhd_kernel[0,0, my_radius, my_radius] = 1.0

            else:
                mu = np.random.rand() 
                sigma = np.random.rand() 
                
                #get_gaussian_kernel(radius=13, mu=0.5, sigma=0.15, r_scale=1.0):
                if mm < (3 * self.internal_channels) // 4:
                    nbhd_kernel = get_gaussian_kernel(radius=my_radius, \
                            mu=mu, sigma=sigma)
                else:
                    nbhd_kernel = get_gaussian_edge_kernel(radius=my_radius, \
                            mu=mu, sigma=sigma, mode=mm % 2)

            if nbhd_kernels is None:
                nbhd_kernels = nbhd_kernel
            else:
                nbhd_kernels = torch.cat([nbhd_kernels, nbhd_kernel], dim=0)

        self.add_neighborhood_kernel(nbhd_kernels)
        self.initialize_neighborhood_layer()

        self.initialize_weight_layer()

        if self.ca_mode == "functional" or self.ca_mode == "neurofunctional":
            print(self.ca_mode)
            for pp in range(self.internal_channels):
                
                if (pp > 8):
                    # close to the edge of chaos, from Lenia papers
                    gen_mu = 0.15 * (1 + np.random.randn() * 0.00001)
                    gen_sigma = 0.015 * (1 + np.random.randn() * 0.0001)

                    if np.random.randint(2):
                        genesis_config = {"name": "Gaussian",\
                                "mu": gen_mu, \
                                "sigma": gen_sigma, \
                                "mode": 1}
                    else:
                        genesis_config = {"name": "DoGaussian",\
                                "mu": gen_mu, \
                                "sigma": gen_sigma, \
                                "mode": 1}

                    per_mu = 0.15 * (1 + np.random.randn() * 0.001) 
                    per_sigma = 0.015 * (1 + np.random.randn() * 0.001) 


                    if np.random.randint(2):
                        persistence_config = {"name": "Gaussian",\
                                "mu": per_mu, \
                                "sigma": per_sigma, \
                                "mode": 1}
                    else:
                        persistence_config = {"name": "DoGaussian",\
                                "mu": per_mu, \
                                "sigma": per_sigma, \
                                "mode": 1}
                    
                else:
                    num_coefficients = 8
                    scaler = torch.tensor(\
                            [10**ii for ii in range(num_coefficients)])

                    per_coefficients = torch.rand(num_coefficients,) / scaler
                    gen_coefficients = torch.rand(num_coefficients,) / scaler

                    genesis_config = {"name": "Polynomial", \
                            "coefficients": gen_coefficients}

                    persistence_config = {"name": "Polynomial", \
                            "coefficients": per_coefficients}

                self.add_genesis_fn(genesis_config)
                self.add_persistence_fn(persistence_config)

        elif self.ca_mode == "neural":
            for pp in range(self.internal_channels):
                genesis_config = {"name": "Identity"}
                persistence_config = {"name": "Identity"}

                self.add_genesis_fn(genesis_config)
                self.add_persistence_fn(persistence_config)
        if (0):    
            for pp in range(self.internal_channels):
                genesis_config = {"name": "ReLU"}
                persistence_config = {"name": "ReLU"}

                self.add_genesis_fn(genesis_config)
                self.add_persistence_fn(persistence_config)

        self.dt = 0.1

        self.include_parameters()
        
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

        self.id_layer = nn.Conv2d(self.external_channels, \
                self.internal_channels, self.id_dim, padding=padding, \
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

        self.neighborhood_layer = nn.Conv2d(self.external_channels, \
                self.internal_channels, \
                self.neighborhood_dim, padding=padding, \
                groups=self.external_channels, \
                padding_mode = self.conv_mode, bias=False)


        for param in self.neighborhood_layer.named_parameters():
            param[1].requires_grad = False
            param[1][:] = self.neighborhood_kernels
        
        
    def initialize_weight_layer(self):

        weights = torch.ones(self.external_channels, self.internal_channels,1,1)

        self.weights = weights / weights.sum()

        if self.ca_mode == "functional":
            self.weights_layer = nn.Conv2d(\
                    self.internal_channels, self.external_channels, 1, \
                    padding=0, \
                    padding_mode = self.conv_mode, bias=False)

            for param in self.weights_layer.named_parameters():
                param[1].requires_grad = False 
                param[1][:] = self.weights
                param[1].requires_grad = self.use_grad
        else:
            
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
        
        if config["name"] == "Gaussian":
            self.genesis_fns.append(Gaussian(**config))
        elif config["name"] == "DoGaussian":
            self.genesis_fns.append(DoGaussian(**config))
        elif config["name"] == "DoubleGaussian":
            self.genesis_fns.append(DoubleGaussian(**config))
        elif config["name"] == "GaussianMixture":
            self.genesis_fns.append(GaussianMixture(**config))
        elif config["name"] == "Polynomial":
            self.genesis_fns.append(Polynomial(**config))   
        elif config["name"] == "ReLU":
            self.genesis_fns.append(nn.ReLU())
        elif config["name"] == "Tanh":
            self.genesis_fns.append(nn.Tanh())
        elif config["name"] == "Identity":
            self.genesis_fns.append(Identity(**config))
        elif config["name"] == "SmoothIntervals":
            self.genesis_fns.append(SmoothIntervals(**config))
        else:
            print(f"warning, fn {config['fn']} not implemented yet")
        
    def add_persistence_fn(self, config):

        if config["name"] == "Gaussian":
            self.persistence_fns.append(Gaussian(**config))
        elif config["name"] == "DoGaussian":
            self.persistence_fns.append(DoGaussian(**config))
        elif config["name"] == "DoubleGaussian":
            self.persistence_fns.append(DoubleGaussian(**config))
        elif config["name"] == "GaussianMixture":
            self.persistence_fns.append(GaussianMixture(**config))
        elif config["name"] == "Polynomial":
            self.persistence_fns.append(Polynomial(**config))
        elif config["name"] == "ReLU":
            self.persistence_fns.append(nn.ReLU())
        elif config["name"] == "Tanh":
            self.persistence_fns.append(nn.Tanh())
        elif config["name"] == "Identity":
            self.persistence_fns.append(Identity(**config))
        elif config["name"] == "SmoothIntervals":
            self.persistence_fns.append(SmoothIntervals(**config))
        else:
            print(f"warning, fn {config['fn']} not implemented yet")
    
    def include_parameters(self):

        for ii, genesis_fn in enumerate(self.genesis_fns):

            for jj, param in enumerate(genesis_fn.parameters()):
                self.register_parameter(f"gen_fn_{ii}_{jj}", param)

        for kk, persistence_fn in enumerate(self.persistence_fns):

            for ll, param in enumerate(persistence_fn.parameters()):
                self.register_parameter(f"per_fn_{kk}_{ll}", param)

    def persistence(self, neighborhoods):
        
        error_msg = f"expected number of neighborhoods and update function " \
                f"equal, got {len(self.persistence_fns)} update functions " \
                f"and {neighborhoods.shape[1]} neighborhoods"
        assert neighborhoods.shape[1] == len(self.persistence_fns), error_msg 

        updated_grid = torch.Tensor().to(self.my_device)

        for ii in range(len(self.persistence_fns)):

            temp_grid = self.persistence_fns[ii](neighborhoods[:,ii:ii+1,:,:]) 

            updated_grid = torch.cat([updated_grid, temp_grid], dim=1)

        return updated_grid
    
    def genesis(self, neighborhoods):
        
        error_msg = f"expected number of neighborhoods and update function " \
                f"to be equal, got {len(self.genesis_fns)} update functions " \
                f"and {neighborhoods.shape[1]} neighborhoods"
        assert neighborhoods.shape[1] == len(self.genesis_fns), error_msg 

        updated_grid = torch.Tensor().to(self.my_device)

        for ii in range(len(self.genesis_fns)):

            temp_grid = self.genesis_fns[ii](neighborhoods[:,ii:ii+1,:,:]) 

            updated_grid = torch.cat([updated_grid, temp_grid], dim=1)

        return updated_grid



    def id_conv(self, universe):
        """
        """
        
        return self.id_layer(universe)
    
    def neighborhood_conv(self, universe):

        return self.neighborhood_layer(universe)

    def alive_mask(self, universe):
        """
        zero out cells not meeting a threshold in the alpha channel
        
        """

        alive_mask = torch.zeros_like(universe[:, 3:4, :, :])

        alive_mask[universe[:, 3:4, :, :] > self.alive_threshold] = 1.0

        return universe * alive_mask

    def update_universe(self, identity, neighborhoods):

        generate = (1.0 - identity) * self.genesis(neighborhoods)
        persist = identity * self.persistence(neighborhoods)

        genper = generate + persist

        update = self.weights_layer(genper)

        return update 

    def forward(self, universe, mode=0):

        if universe.shape[1] >= 4:
            universe = self.alive_mask(universe)

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
        
        batch_size = query_kwargs("batch_size", 8, **kwargs)
        num_ca_steps = query_kwargs("number_ca_steps", 12, **kwargs)
        num_loss_steps = query_kwargs("num_loss_steps", 4, **kwargs)
        lr = query_kwargs("lr", 1e-4, **kwargs)  
        number_display = query_kwargs("updates", 10, **kwargs)
        max_steps = query_kwargs("max_steps", 1000, **kwargs)

        aperture_radius = query_kwargs("aperture_radius", 1.0, **kwargs) 
        bite_radius = query_kwargs("bite_radius", 0.1, **kwargs)
        my_noise = query_kwargs("my_noise", 0.0, **kwargs)
        max_bite_radius = query_kwargs("max_bite_radius", 0.5, **kwargs)
        max_ca_steps = query_kwargs("max_ca_steps", 32, **kwargs)
        loss_threshold = query_kwargs("loss_threshold", 0.1, **kwargs)
        
        self.my_device = query_kwargs("device", "cpu", **kwargs)
        #external_channels = query_kwargs("external_channels", 8, **kwargs)
        #internal_channels = query_kwargs("internal_channels", 32, **kwargs)
        #ca_mode = query_kwargs("ca_mode", "neural" **kwargs)


        self.random_init()
        self.include_parameters()
        self.to(kwargs["device"])
        self.dt = query_kwargs("dt", 0.25, **kwargs)

        target = self.alive_mask(target)

        if (0):
            for hh in range(self.neighborhood_kernels.shape[0]):

                plt.figure()
                plt.imshow(self.neighborhood_kernels[hh,0,:,:].detach())

                plt.savefig(f"./assets/kernel{hh}.png")
    
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # set loss to None for first progress statement
        loss = None

        print(f"batch size, target.shape = {batch_size}, {target.shape}")
        print(f"ca mode = {self.ca_mode}, weights layer = ", self.weights_layer)

        for training_step in range(max_steps):
            
            batch = utils.prep_input(target, \
                    external_channels=self.external_channels, \
                    batch_size=batch_size, \
                    aperture_radius=aperture_radius, \
                    bite_radius=bite_radius, \
                    noise_level=my_noise)

            batch = batch.to(self.my_device)
            batch = self.alive_mask(batch)

            if training_step % (max_steps // number_display) == 0 or \
                    training_step == 1 or training_step == (max_steps-1):

                print(f"loss at step {training_step} = {loss}")

                if 1:
                    grid = utils.prep_input(target, \
                            external_channels=self.external_channels, \
                            batch_size=1, \
                            aperture_radius=aperture_radius, \
                            bite_radius=bite_radius, \
                            noise_level=my_noise)

                    save_fig_sequence(grid, self, \
                            num_steps=num_ca_steps + num_loss_steps, mode=0, \
                            tag=f"tr_step_{training_step}_",\
                            invert_colors=False)

            loss = 0.0

            optimizer.zero_grad()
            for ca_step in range(num_ca_steps):
                batch = self.forward(batch)
                
            for ca_step in range(num_loss_steps):

                batch = self.forward(batch)
                loss += torch.mean(\
                        (target.to(self.my_device) \
                        - batch[:,:target.shape[1],:,:])**2) \
                        / num_loss_steps

            loss.backward()

            optimizer.step()
            
            if loss < loss_threshold / num_loss_steps:

                num_ca_steps = min([num_ca_steps+2, max_ca_steps])
                bite_radius = min([bite_radius*1.05, max_bite_radius])
                aperture_radius = max([aperture_radius*0.98, 0.05])

                print(f"{loss} ca steps = {num_ca_steps}, " \
                        f"bite_radius = {bite_radius} " \
                        f"aperture radius = {aperture_radius}")
                
                grid = utils.prep_input(target, \
                        external_channels=self.external_channels, \
                        batch_size=1, \
                        aperture_radius=aperture_radius, \
                        bite_radius=bite_radius, \
                        noise_level=0.05)

                save_fig_sequence(grid, self, \
                        num_steps=num_ca_steps+num_loss_steps, mode=0, \
                        tag=f"tr_step_{training_step}_", invert_colors=False)

    def to_device(self, my_device):
        """
        overloads the `to` function from nn.Module
        to ensure all parameters get moved
        """
        
        self.to(my_device)
        self.my_device = my_device
        self.id_layer.to(my_device)
        self.neighborhood_layer.to(my_device)
        self.weights_layer.to(my_device)

        # the section below is important (and indeed the reason for
        # overloading the `to` function) in the case that include_parameters
        # has not been called
        for name, param in self.named_parameters():
            if len(param.shape) == 0:
                param = param.reshape(1)
            param[:] = param.to(my_device)

        for ii, genesis_fn in enumerate(self.genesis_fns):
            genesis_fn.to(my_device)

        for kk, persistence_fn in enumerate(self.persistence_fns):
            persistence_fn.to(my_device)


    def get_genesis_params(self):

        params = np.array([])
        for ii, genesis_fn in enumerate(self.genesis_fns):
            for jj, param in enumerate(genesis_fn.named_parameters()):
                params = np.append(params, param[1].detach().cpu().numpy().ravel())

        return params

    def get_persistence_params(self):

        params = np.array([])
        for kk, persistence_fn in enumerate(self.persistence_fns):
            for ll, param in enumerate(persistence_fn.named_parameters()):
                params = np.append(params, param[1].detach().cpu().numpy().ravel())

        return params

    def get_params(self):
    
        params = np.array([])

        if self.ca_mode != "functional": 
            # weights are not a learnable parameter for functional CA
            for hh, param in enumerate(self.weights_layer.named_parameters()):
                params = np.append(params, param[1].detach().cpu().numpy().ravel())

        params = np.append(params, self.get_genesis_params())
        params = np.append(params, self.get_persistence_params())


        return params

    def set_params(self, params):

        self.no_grad()
        param_start = 0
        if self.ca_mode != "functional": 
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

        for ii, genesis_fn in enumerate(self.genesis_fns):
            for jj, param in genesis_fn.named_parameters():

                if not len(param.shape):
                    param = param.reshape(1)

                param_stop = param_start + reduce(lambda x,y: x*y, param.shape)
                param[:] = nn.Parameter( \
                        torch.tensor( \
                        params[param_start:param_stop].reshape(param.shape),\
                        requires_grad = self.use_grad), \
                        requires_grad = self.use_grad)

                param_start = param_stop

        for kk, persistence_fn in enumerate(self.persistence_fns):
            for ll, param in persistence_fn.named_parameters():

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

        for ii, genesis_fn in enumerate(self.genesis_fns):
            for jj, param in enumerate(genesis_fn.parameters()):
                param.requires_grad = False

        for kk, persistence_fn in enumerate(self.persistence_fns):
            for ll, param in enumerate(persistence_fn.parameters()):
                param.requires_grad = False
        

if __name__ == "__main__":


    if (0):

        ca = CA()
        ca.default_init()
        orbium = get_orbium()

        starting_grid = torch.zeros(4,1, 64, 64)
        starting_grid[:,:,:20,:20] = torch.tensor(orbium)
        starting_grid[:,:,:20:,-20:] = torch.rand(4, 1, 20, 20)

        grid = ca(starting_grid)


        save_fig_sequence(grid, ca, num_steps=256, mode=0, cmap=plt.get_cmap("magma"))
        

    elif (1):
        import time

        orbium = get_orbium()
        v_cpu = []
        v_cuda = []
        fps_cpu = []
        fps_cuda = []
        num_steps = 10000

        for my_device in ("cuda", "cpu"):

            print(my_device)
            for vectorization in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                print(vectorization)

                ca = CA()
                ca.default_init()
                ca.to(my_device)
                ca.no_grad()
                
                starting_grid = torch.rand(vectorization, 1, 64, 64)
                starting_grid[:,:,:20,:20] = torch.tensor(orbium)
                grid = starting_grid.to(my_device)

                t0 = time.time()
                for step in range(num_steps):
                    grid = ca(grid)

                t1 = time.time()

                time_elapsed = t1-t0
                fps = (num_steps * vectorization) / time_elapsed


                if my_device == "cpu":
                    v_cpu.append(vectorization)
                    fps_cpu.append(fps)
                else:
                    v_cuda.append(vectorization)
                    fps_cuda.append(fps)


        my_cmap = plt.get_cmap("magma")
        plt.figure()
        plt.plot(v_cpu, fps_cpu, lw=3, color=my_cmap(64), label="cpu")
        plt.plot(v_cuda, fps_cuda, lw=3, color=my_cmap(196), label="cuda")
        plt.legend()
        plt.show()
