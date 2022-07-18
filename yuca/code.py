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


class CODE(CA):

    def __init__(self, **kwargs):
        super(CODE, self).__init__(**kwargs)
    
        self.prev_dt = query_kwargs("prev_dt", 0.3, **kwargs)
        self.max_dt = query_kwargs("max_dt", 0.5, **kwargs)
        self.min_dt = query_kwargs("min_dt", 0.01, **kwargs)
        self.error_threshold = query_kwargs("error_threshold", 1e-3, **kwargs)
        self.static_dt = query_kwargs("static_dt", 1.0, **kwargs)

        self.reset()

    def reset(self):

        self.t_count = 0.0

    def adaptive_euler(self, grid, total_step=0.0):
        
        keep = False
        
        current_step = min([self.prev_dt * 2, self.max_dt])
        # Don't track gradients while estimating step size
        with torch.no_grad():

            while (not keep) and current_step >= self.min_dt:
                self.dt = current_step
                
                big_step_grid = self.get_new_grid(grid)

                self.dt = current_step * 0.5

                small_step_grid = self.get_new_grid(grid)
                small_step_grid = self.get_new_grid(small_step_grid)

                mean_error = torch.abs(big_step_grid - small_step_grid) #**2
                mean_error *= 1.0 * (small_step_grid > 0.0)
                mean_error[mean_error > 0.0] = (mean_error / small_step_grid)[mean_error > 0.0]

                step_mse = mean_error.mean() #.max()
                #torch.max((big_step_grid - small_step_grid)**2)

                if step_mse < self.error_threshold:
                    keep = True
                    self.dt = current_step
                else:
                    current_step = current_step * 0.75



            if not keep:
                #print(f"{step_mse} does not meet error threshold, using min_dt of  {min_dt}")
                current_step = self.min_dt
                self.dt = current_step
            else:
                pass
                #print(f"{step_mse} does meet error threshold, using {ca.dt}")

        self.nmse = step_mse            
        self.t_count += self.dt #min([self.dt, self.static_dt - total_step])
        self.prev_dt = self.dt

        grid = self.get_new_grid(grid)
#        if self.static_dt -total_step > self.dt:
#            total_step += self.dt
#            grid = self.get_new_grid(grid)
#            grid = self.adaptive_euler(grid, total_step=total_step)
#        elif total_step >= self.static_dt:
#            pass
#        else:
#            self.dt = self.static_dt - total_step
#            total_step += self.dt
#            grid = self.get_new_grid(grid)
#            grid = self.adaptive_euler(grid, total_step=total_step)
#            
        return grid
    
    
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
        
        new_universe = torch.clamp(universe + self.dt * update, 0, 1.0)

        return new_universe
    
    def forward(self, universe, mode=0):

        grid = self.adaptive_euler(universe)
        
        return grid

if __name__ == "__main__":

    dim = 64
    ca = CODE(tag="gemini", kernel_radius=15)
    ca.no_grad()

    grid = torch.rand(1, 1, dim, dim)

    plt.figure(); plt.imshow(grid.cpu().numpy().squeeze())
    print(f"starting t_count = {ca.t_count}")
    while ca.t_count <= 10.0:
        grid = ca(grid)


    print(f"ending t_count = {ca.t_count}")

    plt.figure(); plt.imshow(grid.cpu().numpy().squeeze())

    plt.show()

