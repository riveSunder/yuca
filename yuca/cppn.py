import sys
import argparse
import subprocess
import os

from functools import reduce

from random import shuffle
import time
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

from yuca.utils import query_kwargs, seed_all, save_fig_sequence
from yuca.activations import Gaussian
from yuca.params_agent import ParamsAgent

import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

    
class CPPN(nn.Module):

    def __init__(self, **kwargs):

        super(CPPN, self).__init__()

        
        self.internal_channels = query_kwargs("internal_channels", \
                1, **kwargs) 
        self.external_channels = query_kwargs("external_channels", \
                1, **kwargs) 
        self.hidden_dim = query_kwargs("hidden_dim", 16, **kwargs)
        self.grid_dim = query_kwargs("dim", 64, **kwargs)
        self.r_threshold = 0.75

        self.init_model()
        _ = self.get_cppn_input()

        self.no_grad()

    def init_model(self):

        input_dim = self.external_channels * 16

        self.model = nn.Sequential(\
                nn.Linear(input_dim, self.hidden_dim), Gaussian(),\
                nn.Linear(self.hidden_dim, self.hidden_dim), Gaussian(),\
                nn.Linear(self.hidden_dim, self.hidden_dim), Gaussian(),\
                nn.Linear(self.hidden_dim, self.external_channels), Gaussian())


    def get_cppn_input(self):
        
        xx, yy = np.meshgrid(np.arange(-1.0, 1.0 + 2.0 /self.grid_dim,\
                2./(self.grid_dim-1)), \
                np.arange(-1.0, 1.0 + 2.0 / self.grid_dim,\
                2./(self.grid_dim-1)))

        eps = 1e-6

        rr = np.sqrt(xx**2 + yy**2)
        sinr_16r = np.sin(rr*np.pi*16)  / (rr + eps) 
        sinr_8r = np.sin(rr*np.pi*8)  / (rr + eps)
        sinr_4r = np.sin(rr*np.pi*4)  / (rr + eps)
        sinr_16 = np.sin(rr*np.pi*16) 
        sinr_8 = np.sin(rr*np.pi*8) 
        sinr_4 = np.sin(rr*np.pi*4) 

        sinr_16r /= sinr_16r.max()
        sinr_8r /= sinr_8r.max()
        sinr_4r /= sinr_4r.max()

        sinr_16 /= sinr_16.max()
        sinr_8 /= sinr_8.max()
        sinr_4 /= sinr_4.max()


        sinr_16r = torch.tensor(sinr_16r).reshape(\
                1,1,sinr_16r.shape[0], sinr_16r.shape[1])
        sinr_8r = torch.tensor(sinr_8r).reshape(\
                1,1,sinr_8r.shape[0], sinr_8r.shape[1])
        sinr_4r = torch.tensor(sinr_4r).reshape(\
                1,1,sinr_4r.shape[0], sinr_4r.shape[1])

        rr = torch.tensor(rr).reshape(1,1,rr.shape[0], rr.shape[1])
        xx = torch.tensor(xx).reshape(1,1,xx.shape[0], xx.shape[1])
        yy = torch.tensor(yy).reshape(1,1,yy.shape[0], yy.shape[1])
        xx2 = xx**2
        xx3 = xx**3

        
        xx2 /= xx2.max()
        xx3 /= xx3.max()

        my_2x = 2**(xx)
        my_2x /= my_2x.max()

        my_3x = 3**(xx)
        my_3x /= my_3x.max()

        absxx = np.abs(xx)
        absyy = np.abs(yy)

        sinxx = torch.sin(np.pi * xx)
        cosyy = torch.cos(np.pi *yy)

        sin2xx = torch.sin(np.pi *2*xx)
        cos2yy = torch.cos(np.pi *2*yy)

        sin4xx = torch.sin(np.pi *4*xx)
        cos4yy = torch.cos(np.pi *4*yy)

        grid = torch.cat([xx, 1.0 - rr], dim=1) 

        grid = torch.cat([grid, absxx], dim=1) 
        grid = torch.cat([grid, absyy], dim=1) 

        grid = torch.cat([grid, my_2x], dim=1) 

        grid = torch.cat([grid,  xx2], dim=1)
        grid = torch.cat([grid,  xx3], dim=1)

        # periodic functions
        grid = torch.cat([grid,  cosyy], dim=1)
        grid = torch.cat([grid,  sinxx], dim=1)
        
        grid = torch.cat([grid,  cos2yy], dim=1)
        grid = torch.cat([grid,  sin2xx], dim=1)

        grid = torch.cat([grid,  cos4yy], dim=1)
        grid = torch.cat([grid,  sin4xx], dim=1)

        grid = torch.cat([grid,  sinr_4r], dim=1)
        grid = torch.cat([grid,  sinr_8r], dim=1)
        grid = torch.cat([grid,  sinr_16r], dim=1)

        self.grid = grid.to(torch.get_default_dtype())

        self.rr = rr

        return grid.to(torch.get_default_dtype())

    def forward(self, x):

        return self.model(x)

    def get_action(self, grid=None):

        if grid == None: # and it should
            grid = self.grid #get_cppn_input()


        pattern = self.forward(grid.transpose(1,3))
        pattern = pattern.transpose(3,1)

        pattern *= (1.0 - self.rr)**2 
        pattern[self.rr >= self.r_threshold] *= 0.000001
        pattern = F.relu(pattern)

        return pattern

    def get_params(self):
    
        params = np.array([])

        for name, param in self.model.named_parameters():
            params = np.append(params, param.detach().cpu().numpy().ravel())

        return params
        
    def set_params(self, params):

        param_start = 0

        for name, param in self.model.named_parameters():
            if not len(param.shape):
                param = param.reshape(1)


            param_stop = param_start + reduce(lambda x,y: x*y, param.shape) 

            param[:] = nn.Parameter(\
                    torch.tensor(\
                    params[param_start:param_stop].reshape(param.shape)),\
                    requires_grad = False)

            param_start = param_stop

    def to_device(self, device):
        
        self.my_device = torch.device(device)
        
        self.to(self.my_device)

        self.grid = self.grid.to(self.my_device)
        self.rr = self.rr.to(self.my_device)

    def no_grad(self):

        self.use_grad = False

        for hh, param in enumerate(self.model.parameters()):
            param.requires_grad = False
            
class CPPNPlus(CPPN):
    def __init__(self, **kwargs):
        super().__init__()

        self.params_agent = ParamsAgent(**kwargs)

    def get_pattern_action(self, grid=None):
        
        return self.get_action(grid)

    def get_rule_action(self, obs=None):
        
        return self.params_agent.get_action()

    def get_params(self):
    
        params = super().get_params()

        params = np.append(params, self.params_agent.get_params())

        return params
        
    def set_params(self, params):

        param_start = 0

        for name, param in self.model.named_parameters():
            if not len(param.shape):
                param = param.reshape(1)


            param_stop = param_start + reduce(lambda x,y: x*y, param.shape) 

            param[:] = nn.Parameter(\
                    torch.tensor(\
                    params[param_start:param_stop].reshape(param.shape)),\
                    requires_grad = False)

            param_start = param_stop

        self.params_agent.set_params(params[param_start:])

