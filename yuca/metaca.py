import os
import copy

import numpy as np
from functools import reduce

import skimage
import skimage.io as sio

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
my_cmap = plt.get_cmap("magma")

from yuca.ca.continuous import CCA

class MetaCA(CCA):

    def __init__(self, **kwargs):
        super(MetaCA, self).__init__(**kwargs)

        self.ddt = 0.25
        self.reset()

    def forward(self, grid, mode=0):
        

        if grid.shape != self.dgrid.shape:
            self.dgrid = torch.zeros_like(grid)

        if grid.shape[1] >= 4:
            grid = self.alive_mask(grid)

        identity = self.id_conv(grid)
        neighborhoods = self.neighborhood_conv(grid)

        update_update = self.update_universe(identity, neighborhoods)

        self.dgrid = self.dgrid + self.ddt * update_update
        
        new_grid = torch.clamp(grid + self.dt * self.dgrid, 0, 1.0)
        
        return new_grid

    def reset(self):

        self.dgrid = torch.tensor([0])

if __name__ == "__main__":

    dim = 64

    grid = torch.rand(1,1,dim, dim)

    orbium = get_orbium()
    grid[:,:,10:10+orbium.shape[0], 10:10+orbium.shape[1]] = torch.tensor(orbium)

    dgrid = torch.zeros_like(grid)

    ca = MetaCA(tag="gemini")
    ca.no_grad()

    plt.figure()
    plt.subplot(121)
    plt.imshow(grid.detach().cpu().squeeze())
    plt.colorbar()
    plt.title("grid")
    plt.subplot(122)
    plt.imshow(dgrid.detach().cpu().squeeze())
    plt.colorbar()
    plt.title("dgrid")
    plt.suptitle("before")


    for ii in range(1024):
        grid, dgrid = ca(grid, dgrid=dgrid)

        my_img = np.append(grid.squeeze().numpy(), dgrid.squeeze().numpy(), axis=1)

        sio.imsave(f"assets/frames/frame_{ii}.png", np.uint8(255*my_cmap(my_img)))

    plt.figure()
    plt.subplot(121)
    plt.imshow(grid.detach().cpu().squeeze())
    plt.title("grid")
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(dgrid.detach().cpu().squeeze())
    plt.colorbar()
    plt.title("dgrid")
    plt.suptitle("after")
    plt.show()
