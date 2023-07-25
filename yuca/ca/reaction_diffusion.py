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

    def __init__(self, **kwargs):
        super().__init__()
        # kwargs are not used, because this model is rigidly defined
        
        for key in ["internal_channels", "external_channels"]:
            if key in kwargs.keys() and kwargs[key] != 2:
                warning = f"Gray-Scott model allows only exactly 2 channels, but "\
                        f"{kwargs[key]} provided for {key}"

            
        self.internal_channels = 2
        self.external_channels = 2
        self.kernel_radius = 1

        self.default_init()
        self.reset()

    def get_dx(self):

        # self.dx is a torch tensor
        return self.dx.item()

    def get_diffusion_u(self):

        # self.diffusion_v is a torch tensor
        return self.diffusion_u.item()

    def get_diffusion_v(self):

        # self.diffusion_v is a torch tensor
        return self.diffusion_v.item()

    def set_dx(self, new_dx):

        if type(new_dx) is torch.Tensor:
            self.dx = new_dx.to(self.my_device)
        else:
            self.dx = torch.tensor(new_dx).to(self.my_device)

    def set_diffusion_u(self, new_diffusion_u):

        if type(new_diffusion_u) is torch.Tensor:
            self.diffusion_u = new_diffusion_u.to(self.my_device)
        else:
            self.diffusion_u = torch.tensor(new_diffusion_u).to(self.my_device)

    def set_diffusion_v(self, new_diffusion_v):

        if type(new_diffusion_v) is torch.Tensor:
            self.diffusion_v = new_diffusion_v.to(self.my_device)
        else:
            self.diffusion_v = torch.tensor(new_diffusion_v).to(self.my_device)

    def load_config(self, config, verbose=False):

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

        if "dt" in config.keys():
            self.dt = torch.tensor(config["dt"])
        else: 
            self.dt = torch.tensor([0.5])

        if "dx" in config.keys():
            self.dx = torch.tensor(config["dx"])
        else: 
            self.dx = torch.tensor([1./330.])

        if "decay_rate" in config.keys():
            self.k = torch.tensor(config["decay_rate"])
        else: 
            self.k = torch.tensor([0.00609])

        if "feed_rate" in config.keys():
            self.f = torch.tensor(config["feed_rate"])
        else: 
            self.f = torch.tensor([0.00620])

        if "diffusion_u" in config.keys():
            self.diffusion_u = torch.tensor(config["diffusion_u"])
        else: 
            self.diffusion_u = torch.tensor([2e-5])

        if "diffusion_v" in config.keys():
            self.diffusion_v = torch.tensor(config["diffusion_v"])
        else: 
            self.diffusion_v = torch.tensor([1e-5])

        self.set_params(config["params"])
        self.include_parameters()

    def restore_config(self, filepath, verbose=False):
        if "\n" in filepath:
            filepath = filepath.replace("\n","")

        file_directory = os.path.abspath(__file__).split("/")
        root_directory = os.path.join(*file_directory[:-3])
        default_directory = os.path.join("/", root_directory, "ca_configs")

        if os.path.exists(filepath):

            config = np.load(filepath, allow_pickle=True).reshape(1)[0]
            self.load_config(config)
            if verbose:
                print(f"config restored from {filepath}")

        elif os.path.exists(os.path.join(default_directory, filepath)):
            
            filepath = os.path.join(default_directory, filepath)
            config = np.load(filepath, allow_pickle=True).reshape(1)[0]
            self.load_config(config)
            if verbose:
                print(f"config restored from {filepath}")
        
        else:

            print(f"default directory: {default_directory}")
            print(f"attempted to read {filepath}, not found")
            #assert False, f"{filepath} not found"
        
    def make_config(self, verbose=False):

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
            neighborhood_kernel_config = {}
            neighborhood_kernel_config["name"] = "LaplacianOfGaussian"
            neighborhood_kernel_config["kernel_kwargs"] = {}
            neighborhood_kernel_config["radius"] = self.kernel_radius
        else:
            neighborhood_kernel_config = self.neighborhood_kernel_config

        # nca params
        config["params"] = self.get_params()
            
        config["id_kernel_config"] = id_kernel_config
        config["neighborhood_kernel_config"] = neighborhood_kernel_config

        config["dt"] = self.dt.item() 
        config["dx"] = self.dx.item() 
        config["diffusion_u"] = self.diffusion_u.item()
        config["diffusion_v"] = self.diffusion_v.item()
        config["feed_rate"] = self.f.item()
        config["decay_rate"] = self.k.item()

        return copy.deepcopy(config)

    def save_config(self, filepath, config=None):

        if config is None:
            config = self.make_config()

        np.save(filepath, config)

    def default_init(self):

        self.neighborhood_kernel_config = {}
        self.neighborhood_kernel_config["name"] = "LaplacianOfGaussian"
        self.neighborhood_kernel_config["kernel_kwargs"] = {"sigma": 0.5}
        self.neighborhood_kernel_config["radius"] = self.kernel_radius

        kernel = get_kernel(self.neighborhood_kernel_config)

        self.add_neighborhood_kernel(kernel)
        self.initialize_neighborhood_layer()

        self.add_identity_kernel()
        self.initialize_id_layer()

    
        """
        Gray-Scott model 
        updates according to 

        $\frac{\partial u}{\partial t} = diffusion_u \nabla^2 u - uv^2 + f(1-u)$

        $\frac{\partial v}{\partial t} = diffusion_v \nabla^2 v + uv^2 - (f + k)v$

        """

        # Gray-Scott parameters
        # nominal U-Skate world from Tim Hutton and mrob
        self.f = torch.tensor([0.0620])
        self.k = torch.tensor([0.0609])
        self.diffusion_u = torch.tensor([2e-5])
        self.diffusion_v = torch.tensor([1e-5])
        # time step
        self.dt = torch.tensor([0.5])
        # spatial step 
        # R.P. Munafo used 1/143. for dx, but I had to adjust to use scalable Laplacian of Gaussian kernels 
        self.dx = torch.tensor([1/330.])


    def add_neighborhood_kernel(self, kernel=None):
        """
        add Laplacian kernel (kernel arg is ignored).
        """

        if kernel is None:
            # fall back to 9-point stencil finite difference kernel
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

    def initialize_grid(self, batch_size=1, dim=256):
        """
        initialize a grid with h3 background equilibrium for u an v
        
        from R.P. Munafo 2014:

        $A = \frac{\sqrt{F}}{(F+k)}$

        and the concentration of $u$ is 

        $u_{h3} = \frac{A - \sqrt{A^2 - 4}}{2A} $

        and $v$

        $v_{h3} = \frac{\sqrt{F} (A +  \sqrt{A^2 - 4}) }{3}$


        batch_size: int - number of samples in a batch
        dim: int or 2-tuple - row and column dimensions for grid
        """

        if type(dim) is int:
            dim = [dim, dim]

        grid = torch.zeros(batch_size, self.external_channels, dim[0], dim[1])

        A = self.f.sqrt() / (self.f + self.k)
        uh3 = (A - torch.sqrt(A**2 - 4)) / (2*A)
        vh3 = (torch.sqrt(self.f) * (A + torch.sqrt(A**2 - 4))) / 2

        grid[:,0,:,:] = uh3
        grid[:,1,:,:] = vh3

        return grid

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
        updates according to 

        $\frac{\partial u}{\partial t} = diffusion_u \nabla^2 u - uv^2 + f(1-u)$

        $\frac{\partial v}{\partial t} = diffusion_v \nabla^2 v + uv^2 - (f + k)v$

        args:
            identity - current state
            neighborhoods - laplacian of current state

        both identity and neighborhoods have dims of NxCxHxW, where C (channels) is 2,
        representing the values for species u and species v
        """

        update = 0 * identity

        nabla_u = neighborhoods[:,0,:,:] / self.dx**2
        nabla_v = neighborhoods[:,1,:,:] / self.dx**2 
        u = identity[:,0,:,:]
        v = identity[:,1,:,:]

        # species u
        update[:,0,:,:] = self.diffusion_u * nabla_u - u*v*v + self.f*(1-u)
        # species v
        update[:,1,:,:] = self.diffusion_v * nabla_v + u*v*v - (self.f+self.k)*v

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
        
        new_universe = universe + self.dt * update 
        new_universe = torch.clamp(new_universe,0,1.)

        self.t_count += self.dt.detach().cpu()

        return new_universe


    def get_params(self):

        params = np.array([])

        for param in [self.diffusion_u, self.diffusion_v, self.f, self.k]:
            params = np.append(params, param.detach().cpu().numpy().ravel())

        return params

    def set_params(self, params):
        self.no_grad()

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
            self.register_parameter(f"rd_{jj}", torch.nn.Parameter(param))

    def no_grad(self):

        self.use_grad = False

        for hh, param in enumerate(self.parameters()):
            param.requires_grad = False

    def to_device(self, my_device):
        """
        additional functionality beyong `to` function from nn.Module
        to ensure all parameters get moved for CCA
        """
        
        self.no_grad()
        self.to(my_device)
        self.my_device = my_device
        self.id_layer.to(my_device)
        self.neighborhood_layer.to(my_device)

        self.dx = self.dx.to(my_device)
        self.dt = self.dt.to(my_device)

        self.diffusion_u = self.diffusion_u.to(my_device)
        self.diffusion_v = self.diffusion_v.to(my_device)
        self.f = self.f.to(my_device)
        self.k = self.k.to(my_device)
        self.no_grad()


