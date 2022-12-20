import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

import yuca

from yuca.ca.continuous import CCA
from yuca.ca.reaction_diffusion import RxnDfn
from yuca.ca.neural import NCA

from yuca.wrappers.halting_wrapper import SimpleHaltingWrapper,\
        HaltingWrapper
from yuca.wrappers.glider_wrapper import GliderWrapper

from yuca.utils import query_kwargs, get_bite_mask, save_fig_sequence

WRAPPER_DICT = {"SimpleHaltingWrapper": SimpleHaltingWrapper,\
        "HaltingWrapper": HaltingWrapper,\
        "GliderWrapper": GliderWrapper}

class CoevolutionWrapper():
    """
    Combine two reward wrappers
    to co-evolve parameters 
    """

    def __init__(self, use_grad=False, **kwargs):
        
        self.ca_fn = query_kwargs("ca_fn", CCA, **kwargs)

        # rule wrappers should be first
        self.wrappers = query_kwargs("wrappers", ["SimpleHaltingWrapper",\
            "GliderWrapper"], **kwargs)

        self.wrappers = [WRAPPER_DICT[elem](**kwargs) for elem in self.wrappers]

        self.my_device = query_kwargs("device", "cpu", **kwargs)

        # TODO: set to False for multi-objective optimization
        self.reduce_fitness = np.mean
        self.action_space = self.ActionSpace(coevo=self)

    class ActionSpace():

        def __init__(self, coevo):

            self.shape = []
            self.coevo = coevo

            for nn in range(len(self.coevo.wrappers)):
                self.shape.append(self.coevo.wrappers[nn].action_space.shape)
                
        def sample(self):

            action = []
            for oo in range(len(self.coevo.wrappers)):
                action.append(self.coevo.wrappers[oo].action_space.sample())

            return action

    def step(self, action):

        assert len(action) == len(self.wrappers)

        fitness_list = []
        done = False
        info = {}
        
        for ii in range(len(self.wrappers)):

            obs, reward, done, info = self.wrappers[ii].step(action[ii])
            
            fitness_list.append(reward.detach())


            if (ii+1) < len(self.wrappers):
                self.wrappers[ii+1].ca.load_config(self.wrappers[ii].ca.make_config())


        if self.reduce_fitness:
            fitness = self.reduce_fitness(fitness_list)

        return 0, fitness, done, info
