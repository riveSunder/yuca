import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

import yuca
from yuca.multiverse import CA
from yuca.utils import query_kwargs, get_bite_mask, save_fig_sequence


class ParamsAgent():

    def __init__(self, **kwargs):
        super(ParamsAgent, self).__init__()

        #default_params = np.array([0.15, 0.035, 0.325, 0.015, \
        #        0.2, .015, 0.295, 0.015]) 

        default_params = np.array([0.15, 0.015, 0.15, 0.020]) #, \
        #        0.155, 0.008, 0.155, 0.008]) 

        self.params = query_kwargs("params", default_params, 
                **kwargs)
        self.num_params = self.params.shape[0]

        self.filter_params = True #query_kwargs("filter_params", True, **kwargs)

    def get_action(self, obs=None):

        
        if self.filter_params:
            # filters for evolving universes with gaussian update functions
            self.params = np.abs(self.params)

        return self.params

    def get_params(self):

        return self.params

    def set_params(self, params):

        if self.filter_params:
            # filters for evolving universes with gaussian update functions
            params = np.abs(params)

            # prevents rulesets with 0-neighbor cells becoming activate
            
            params = np.clip(params, 0.0001, 1.0)
            #params[2] = np.clip(params[2], 0.005, 1.0)

        self.params = params

    def to_device(self, device):

        # placeholder. This agent doesn't use a device (all params in numpy)
        pass
