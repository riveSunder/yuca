import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

from yuca.multiverse import CA
from yuca.utils import query_kwargs, get_bite_mask

class SimpleHaltingWrapper(nn.Module):

    def __init__(self, **kwargs):
        super(SimpleHaltingWrapper, self).__init__()

        self.ca = CA(**kwargs)
        self.num_blocks = query_kwargs("num_blocks", 4, **kwargs)

        self.ca_steps = query_kwargs("ca_steps", 1024, **kwargs)
        self.batch_size = query_kwargs("batch_size", 8, **kwargs)

        self.dim = query_kwargs("dim", 128, **kwargs)
        self.alive_target = 0.5001

        self.my_device = query_kwargs("device", "cpu", **kwargs)

        #Prediction mode: 0 - vanishing, 1 - static, 2 - both
        self.prediction_mode = query_kwargs("prediction_mode", 0, **kwargs)



    def reset(self):
        pass 

    def eval(self):
        pass


    def train(self):
        pass

    def step(self, action):
    
        self.ca.set_params(action)


        grid = torch.rand(self.batch_size, self.ca.external_channels, \
                self.dim, self.dim).to(self.ca.my_device)
            
        for kk in range(self.ca_steps): 

            grid = self.ca(grid)
            if grid.mean() <= 0.0:
                break

        target = 1.0 * \
                (grid.mean(dim=(2,3)).mean(dim=-1, keepdim=True) > 0.0000)


        if grid.mean() == 0.0 and target.mean():
            print(f"warning, grid.mean() = {grid.mean()}"\
                    f" and target.mean() = {target.mean()}")

        active_grid = target.mean()
        
        
        done = False
        info = {"active_grid": active_grid}
        info["params"] = action
        info["ca_steps"] = self.ca_steps

        halting_mse = - ((self.alive_target - active_grid)**2).mean()
        reward = halting_mse

        return 0, reward, done, info

    def to_device(self, device):

        self.ca.to_device(device)
        self.my_device = device

    def get_params(self):

        return self.ca.get_params()

    def set_params(self, params):
        
        self.ca.set_params(params)

class HaltingWrapper(nn.Module):

    def __init__(self, **kwargs):
        super(HaltingWrapper, self).__init__()

        self.ca = CA(**kwargs)
        self.num_blocks = query_kwargs("num_blocks", 4, **kwargs)

        self.dropout = query_kwargs("dropout", 0.0, **kwargs)
        self.lr = query_kwargs("lr", 1e-3, **kwargs)
        self.ca_steps = query_kwargs("ca_steps", 1024, **kwargs)
        self.batch_size = query_kwargs("batch_size", 8, **kwargs)

        self.train_steps = 8
        self.dim = query_kwargs("dim", 128, **kwargs)
        self.alive_target = 0.5001

        self.my_device = query_kwargs("device", "cpu", **kwargs)
        #Prediction mode: 0 - vanishing, 1 - static, 2 - both
        self.prediction_mode = query_kwargs("prediction_mode", 0, **kwargs)
        self.bigbang_steps = 8


    def forward(self, x):
        
        for hh in range(len(self.layers)):
            
            x = self.layers[hh](x)
            x = F.avg_pool2d(x, kernel_size=2)

        x = torch.sigmoid(self.final_layer(x))
        
        x = torch.mean(x, dim=(2,3))

        return x

    def reset(self):
        self.init_model()

    def init_model(self):
        """

        """
        self.layers = []
        for ii in range(self.num_blocks):
            self.layers.append(nn.Sequential(\
                    nn.Conv2d(self.ca.external_channels, 32, 3, padding=1), \
                    nn.Dropout(p=self.dropout), \
                    nn.ReLU(), \
                    nn.Conv2d(32, 32, 3, padding=1), \
                    nn.Dropout(p=self.dropout), \
                    nn.ReLU(), \
                    nn.Conv2d(32, self.ca.external_channels, 3, padding=1), \
                    nn.Dropout(p=self.dropout), \
                    nn.ReLU(), \
                    ))
            for name, param in self.layers[ii].named_parameters():
                self.register_parameter(\
                        f"{name.replace('.','')}_layer{ii}", param)

        if (self.prediction_mode == 2):
            #Predict both vanishing and end-run dynamism

            self.final_layer = nn.Conv2d(self.ca.external_channels, \
                    2, 3, padding=1)
        else:
            self.final_layer = nn.Conv2d(self.ca.external_channels, \
                    1, 3, padding=1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.BCELoss()

    def eval(self):

        self.final_layer.eval()
        for jj in range(len(self.layers)):
            self.layers[jj].eval()

    def train(self):

        self.final_layer.train()
        for jj in range(len(self.layers)):
            self.layers[jj].train()

    def get_accuracy(self, prediction, target):

        labels = torch.round(target)
        guesses = torch.round(prediction)

        labels = torch.clamp(labels, 0, 1)
        guesses = torch.clamp(guesses, 0, 1)

        accuracy = (1.0 * (labels == guesses)).mean()
        
        return accuracy

    def step(self, action):
    
        #import pdb; pdb.set_trace()
        self.ca.set_params(action)

        for train_step in range(self.train_steps):
            self.train()

            self.zero_grad()

            grid = torch.rand(self.batch_size, self.ca.external_channels, \
                    self.dim, self.dim).to(self.ca.my_device)
            #grid[:,:,:32,:32] = 1.0

            for mm in range(self.bigbang_steps):
                
                grid = self.ca(grid)


            pred = self.forward(grid)
            
            kk = 0
            while kk < self.ca_steps and grid.mean() > 0.0:

                grid = self.ca(grid)
                kk += 1

            dgrid = torch.abs(grid - self.ca(grid))
            if (self.prediction_mode == 0):
                # predict vanishing
                target = 1.0 * \
                        (grid.mean(dim=(2,3)).mean(dim=-1, keepdim=True) > 0.0001)

            elif (self.prediction_mode == 1):
                # predict end-run dynamisim
                target = 1.0 * \
                        (dgrid.mean(dim=(2,3)).mean(dim=-1, keepdim=True) > 0.0001)

            else:
                # predict vanishing and end-run dynamism
                target_0 = 1.0 * \
                        (grid.max(dim = -1)[0].max(dim = -1)[0].max(\
                        dim = -1, keepdim = True)[0] > 0.00001)
                target_1 = 1.0 * \
                        (dgrid.mean(dim=(2,3)).mean(dim=-1, keepdim=True) > 0.0001)
                target = torch.cat([target_0, target_1], dim=-1)

            loss = self.loss_fn(pred, target)
            loss.backward()
            self.optimizer.step()

        self.eval()

        grid = torch.rand(self.batch_size, self.ca.external_channels, \
                self.dim, self.dim, device=self.my_device)
        #grid[:,:,:32,:32] = 1.0

        for mm in range(self.bigbang_steps):
            grid = self.ca(grid)

        pred = self.forward(grid)

        for kk in range(self.ca_steps):

            grid = self.ca(grid)

            if grid.max() == 0.0:
                break

        dgrid = torch.abs(grid - self.ca(grid))
        if (self.prediction_mode == 0):
            # predict vanishing
            target = 1.0 * \
                    (grid.max(dim = -1)[0].max(dim = -1)[0].max(\
                    dim = -1, keepdim=True)[0] > 0.00001)

        elif (self.prediction_mode == 1):
            # predict end-run dynamisim
            target = 1.0 * \
                    (dgrid.max(dim = -1)[0].max(dim = -1)[0].max(\
                    dim = -1, keepdim=True)[0] > 0.00001)

        else:
            # predict vanishing and end-run dynamism
            target_0 = 1.0 * \
                    (grid.max(dim = -1)[0].max(dim = -1)[0].max(\
                    dim = -1, keepdim=True)[0] > 0.00001)
            target_1 = 1.0 * \
                    (dgrid.max(dim = -1)[0].max(dim = -1)[0].max(\
                    dim = -1, keepdim=True)[0] > 0.00001)
            target = torch.cat([target_0, target_1], dim=-1)

        loss = self.loss_fn(pred, target)
        #reward = loss.detach().cpu()


        reward = - self.get_accuracy(pred, target)

        #An additional reward bonus sets a target proportion of patterns \
        #that don't die out
        active_grid = target.mean()
        
        
        done = False
        info = {"active_grid": active_grid}
        info["predictor_loss"] = reward
        info["params"] = action
        info["ca_steps"] = self.ca_steps

        return 0, reward, done, info

    def to_device(self, device):

        self.ca.to_device(device)
        self.final_layer.to(device)
        self.my_device = device

        for hh in range(len(self.layers)):
            
            self.layers[hh].to(device)

    def get_params(self):

        return self.ca.get_params()

    def set_params(self, params):
        
        self.ca.set_params(params)

