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

from yuca.ca.continuous import CCA


class RandomStepCA(CCA):

    def __init__(self, **kwargs):
        super(RandomStepCA, self).__init__(**kwargs)
    
        self.prev_dt = 0.3
        self.max_dt = 0.25
        self.min_dt = 0.05
        self.error_threshold = 1e-3

        self.reset()

    def reset(self):

        self.t_count = 0.0

    
    def update_universe(self, identity, neighborhoods):

        generate = (1.0 - identity) * self.genesis(neighborhoods)
        persist = identity * self.persistence(neighborhoods)

        genper = generate + persist

        update = self.weights_layer(genper)

        return update 

    def get_new_grid(self, universe):

        if universe.shape[1] >= 4:
            universe = self.alive_mask(universe)

        identity = self.id_conv(universe)
        neighborhoods = self.neighborhood_conv(universe)

        update = self.update_universe(identity, neighborhoods)
        
        self.dt = np.random.rand() * (self.max_dt - self.min_dt) + self.min_dt

        new_universe = torch.clamp(universe + self.dt * update, 0, 1.0)

        self.t_count += self.dt

        return new_universe, 1.0 * self.dt
    
    def forward(self, universe, mode=0):

        grid, self.prev_dt = self.get_new_grid(universe)
        
        return grid

if __name__ == "__main__":

    dim = 64
    ca = RandomStepCA(tag="gemini", kernel_radius=15)
    ca.no_grad()

    grid = torch.rand(1, 1, dim, dim)

    plt.figure(); plt.imshow(grid.cpu().numpy().squeeze())
    print(f"starting t_count = {ca.t_count}")
    while ca.t_count <= 10.0:
        grid = ca(grid)


    print(f"ending t_count = {ca.t_count}")

    plt.figure(); plt.imshow(grid.cpu().numpy().squeeze())

    plt.show()

