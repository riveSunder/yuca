import copy

import torch

def get_orbium_config(radius=13):

    config = {}

    kernel_config = {}
    kernel_config["name"] = "Gaussian"
    kernel_config["radius"] = radius
    kernel_config["kernel_kwargs"] = {"mu": 0.5, "sigma": 0.15}

    gen_config = {}
    gen_config["name"] = "GaussianMixture"
    gen_config["parameters"] = torch.tensor([0.15, 0.015])
    gen_config["mode"] = 1

    config["neighborhood_kernel_config"] = kernel_config 
    config["genesis_config"] = gen_config
    config["persistence_config"] = copy.deepcopy(gen_config)

    return config

def get_geminium_config(radius=18):

    config = {}

    kernel_config = {}
    kernel_config["name"] = "GaussianMixture"
    kernel_config["radius"] = radius
    # parameters for 3 thin concentric rings w/ different weights
    kernel_config["kernel_kwargs"] = {"parameters": torch.tensor([\
            0.5, 0.093809, 0.033,\
            1.0, 0.28143, 0.033, \
            2/3.0, 0.46904, 0.033])}
    

    gen_config = {}
    gen_config["name"] = "GaussianMixture"
    gen_config["parameters"] = torch.tensor([0.26, 0.036])
    gen_config["mode"] = 1

    config["neighborhood_kernel_config"] = kernel_config 
    config["genesis_config"] = gen_config
    config["persistence_config"] = copy.deepcopy(gen_config)

    return config

