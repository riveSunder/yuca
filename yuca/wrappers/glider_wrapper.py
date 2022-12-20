import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

import yuca
from yuca.ca.continuous import CCA
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
#        global y_displacement_0
        y_displacement_0 = torch.tensor(0.0) #None 
        #(self.y_grid * action).sum() / (eps + action.sum())

        old_grid = 1.0  * action
        score_every = max([self.ca_steps // 8 , 1])
        self.ca.reset()
        for step in range(self.ca_steps):
       
            action = self.ca(action)

            mean_grid = action.mean()
            max_grid = action.max()

            if step % (score_every) == 0:# or step == (self.ca_steps - 1):
                

#                global y_displacement_0
                y_displacement_1 = (self.y_grid * action).sum() / (eps + action.sum())

                # penalize growth/decay, i.e. incentivize morphological homeostasis
                #
                growth = 1.0 - (mean_grid / (mean_grid_0 + eps))
                mean_grid_0 = mean_grid
                mean_grid = action.mean()

                if y_displacement_0 != 0.0: #None:
                    y_displacement = torch.abs(y_displacement_1 - y_displacement_0)
                    reward += torch.abs(y_displacement.cpu()) - torch.abs(growth.cpu()) * 150
                else: 
                    y_displacement = torch.tensor(0.0)
                    reward += torch.abs(y_displacement.cpu()) - torch.abs(growth.cpu()) * 150

                y_displacement_0 = y_displacement_1.clone()

            dgrid = action - old_grid
            old_grid = 1.0  * action


            if mean_grid == 0.0 or dgrid.max() <= 0.00001:
                break

        if step == 0:
            # empty patterns are extremely penalized 
            reward -= 10000
        if mean_grid == 0.0 or dgrid.max() <= 0.00001:
            # patterns that die out are substantially penalized 
            reward -= 100
        
        reward -= (self.ca_steps - step)
        info["y_displacment"] = y_displacement
        info["growth"] = growth
        info["active_grid"] = mean_grid
        
        return obs, reward, done, info

    def init_displacement_grid(self):
        # displacement matrices
        yy, xx = np.meshgrid(np.arange(self.dim), np.arange(self.dim))

        self.x_grid = torch.tensor(xx).reshape(1, 1, self.dim, self.dim)
        self.y_grid = torch.tensor(yy).reshape(1, 1, self.dim, self.dim)

    def reset(self):
        pass

    def to_device(self, device):
        
        self.my_device = torch.device(device)

        self.x_grid = self.x_grid.to(self.my_device)
        self.y_grid = self.y_grid.to(self.my_device)


