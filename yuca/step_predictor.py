import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

import yuca
from yuca.multiverse import CA
from yuca.utils import query_kwargs, get_bite_mask, save_fig_sequence



class StepPredictor(nn.Module):

    def __init__(self, **kwargs):
        super(StepPredictor, self).__init__()

        self.layer_repeats = query_kwargs("layer_repeats", 1, **kwargs)
        self.kernel_radius = query_kwargs("kernel_radius", 13, **kwargs)
        self.hidden_channels = query_kwargs("hidden_channels", 8, **kwargs)
        self.out_channels = query_kwargs("out_channels", 1, **kwargs)

        self.init_model()
        
    def init_model(self):

        my_padding = self.kernel_radius

        self.conv_layers = []
        

        for ii in range(self.layer_repeats):

            self.conv_layers.append(nn.Conv2d(in_channels=self.out_channels, \
                    out_channels=self.hidden_channels,\
                    kernel_size=2 * self.kernel_radius+1, stride=1, \
                    padding=self.kernel_radius))

            for gg, param in enumerate(self.conv_layers[-1].parameters()):
                self.register_parameter(f"layer{ii}_param{gg}", param)

            self.conv_layers.append(nn.Conv2d(in_channels=self.hidden_channels,\
                    out_channels=self.out_channels, kernel_size=1))

            for hh, param in enumerate(self.conv_layers[-1].parameters()):
                self.register_parameter(f"layer{ii}_1x1_param{hh}", param)
                    
    def forward(self, x):

        for jj in range(0, self.layer_repeats, 2):
            
            x0 = torch.relu(self.conv_layers[jj](x))

            x = x + torch.sigmoid(self.conv_layers[jj + 1](x0))

        return x

            

        
