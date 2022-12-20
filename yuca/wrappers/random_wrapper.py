import time
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

from yuca.ca.continuous import CCA
from yuca.utils import query_kwargs, get_bite_mask

from yuca.wrappers.halting_wrapper import SimpleHaltingWrapper
from yuca.utils import seed_all

class RandomWrapper(SimpleHaltingWrapper):

    def __init__(self, **kwargs):
        
        super(RandomWrapper, self).__init__(**kwargs)

    def step(self, action):

        
        seed_all(int((time.time() % 1)*1000))
        info = {"active_grid": torch.tensor([-1.0])}
        return 0, torch.rand(1), 0, info
        
