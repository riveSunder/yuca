import numpy as np
import time

import torch 
import torch.nn as nn
import torch.nn.functional as F

import yuca

from yuca.ca.continuous import CCA
from yuca.ca.reaction_diffusion import RxnDfn
from yuca.ca.neural import NCA

from yuca.utils import query_kwargs, get_bite_mask, save_fig_sequence


class GliderWrapper():

    def __init__(self, **kwargs):

        ca_fn = query_kwargs("ca_fn", CCA, **kwargs)

        self.ca = ca_fn(**kwargs)
        self.ca_steps = query_kwargs("ca_steps", 1024, **kwargs)
        self.batch_size = query_kwargs("batch_size", 8, **kwargs)
        self.dim = query_kwargs("dim", 128, **kwargs)
        self.my_device = query_kwargs("device", "cpu", **kwargs)

        self.external_channels = query_kwargs("external_channels", 1, **kwargs)

        self.input_filepath = query_kwargs("input_filepath", None, **kwargs)
        self.ca_config = query_kwargs("ca_config", None, **kwargs)

        if self.ca_config is not None:

            self.ca.restore_config(self.ca_config)

        elif self.input_filepath is not None:

            my_data = np.load(self.input_filepath, allow_pickle=True).reshape(1)[0]
            my_params = my_data["elite_params"][-1][0]
            self.ca.set_params(my_params)

        else:
            self.ca.no_grad()

        self.init_displacement_grid()

        self.action_space = self.ActionSpace(\
                shape=(self.batch_size, self.ca.external_channels, self.dim, self.dim))

    class ActionSpace():

        def __init__(self, shape=(1,1,64,64)):

            self.shape = shape

        def sample(self):

            return torch.rand(*self.shape, dtype=torch.get_default_dtype())

    def eval(self):

        self.ca.eval()

    def step(self, action):
        
        return self.meta_step(action)

    def meta_step(self, action):
        
        info = {} 
        reward = torch.tensor(0.0)
        done = False
        obs = 0
        eps = 1e-6

        step = 0
        mean_grid_0 = action.mean()
        mean_grid = action.mean()
        max_grid = 0.31 #action.max()

        # avoid the "tree falling over" hack 
        y_displacement_0 = torch.tensor(0.0) #None 

        old_grid = 1.0  * action
        score_every = 1 #max([self.ca_steps // 8 , 1])
        self.ca.reset()
        kernel_span = self.y_grid.max()
        
        for step in range(self.ca_steps):
       
            action = self.ca(action)

            mean_grid = action.mean()
            max_grid = action.max()

            if step % (score_every) == 0:# or step == (self.ca_steps - 1):
                

                y_displacement_1 = (self.y_grid * action[0:1]).sum() / (eps + action.sum())


                # penalize growth/decay, i.e. incentivize morphological homeostasis
                growth =  torch.abs(1.0 - (mean_grid / (mean_grid_0 + eps)))
                mean_grid_0 = 1.0 * mean_grid

                if y_displacement_0 != 0.0: #None:

                    y_displacement = y_displacement_1 - y_displacement_0

                    if y_displacement_0 > (kernel_span - 2) and y_displacement_1 < 2:
                        # account for wraparound errors
                        y_displacement += kernel_span

                    reward += y_displacement.cpu() - growth.cpu() #* 150
                else: 
                    y_displacement = torch.tensor(0.0)
                    reward += y_displacement.cpu() - growth.cpu() #* 150

                y_displacement_0 = y_displacement_1.clone()


            dgrid = action - old_grid
            old_grid = 1.0  * action

            if mean_grid == 0.0 or dgrid.max() <= 0.00001:
                break

        # take stepwise average 
#        reward /= (step + 1e-9)
#        reward -= (self.ca_steps - step - 1)
        info["y_displacment"] = y_displacement
        info["growth"] = growth
        info["active_grid"] = mean_grid
        
        return obs, reward, done, info

    def init_displacement_grid(self):
        # displacement matrices
        yy, xx = np.meshgrid(np.arange(self.dim), np.arange(self.dim))

        self.x_grid = torch.tensor(xx).reshape(1, 1, self.dim, self.dim)
        self.y_grid = torch.tensor(yy).reshape(1, 1, self.dim, self.dim)

        self.x_grid = self.x_grid / self.ca.kernel_radius
        self.y_grid = self.y_grid / self.ca.kernel_radius

    def reset(self):
        pass

    def to_device(self, device):
        
        self.my_device = torch.device(device)

        self.x_grid = self.x_grid.to(self.my_device)
        self.y_grid = self.y_grid.to(self.my_device)


