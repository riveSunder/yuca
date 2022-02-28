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

from yuca.multiverse import CA


import matplotlib.pyplot as plt

class Lenia(CA):
    """
    A CA framework called Lenia (Chan 2019, Chan 2020)
    """

    def __init__(self, **kwargs):
        super(Lenia, self).__init__(**kwargs)

        self.reset()

    def reset(self):
        
        self.t_count = 0.0
    def update_universe(self, identity, neighborhoods):

        growth = self.genesis(neighborhoods)
        update = self.weights_layer(growth)

        return update 


if __name__ == "__main__":


    if (1):

        ca = Lenia(tag="orbium")
        orbium = get_orbium()

        starting_grid = torch.zeros(4,1, 64, 64)
        starting_grid[:,:,:20,:20] = torch.tensor(orbium)
        starting_grid[:,:,:20:,-20:] = torch.rand(4, 1, 20, 20)

        grid = ca(starting_grid)

        save_fig_sequence(grid, ca, num_steps=256, mode=0, \
                cmap=plt.get_cmap("magma"), tag="test_lenia")
        

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
