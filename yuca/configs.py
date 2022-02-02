import copy

import torch

# functions for building configs for Life-like CA
def get_smooth_interval(rules_list):
    """
    Convert a single list of Life-like CA rules to a set of intervals.
    Converts one list of rules at a time; 
    birth and survival rules must be each converted in turn.
    """

    b = [[(2*bb-1)/16., (2*bb+1)/16.] for bb in rules_list]

    return b

def get_smooth_life_config(radius=10, birth_intervals=None, survival_intervals=None):
    """
    Return a glaberish config based on a Life-like CA's B/S rules.
    The default radius of 1 corresponds to a 3x3 Moore neighborhood. 
    """

    if birth_intervals is None:
        birth_intervals = [[0.278, 0.365]]
    if survival_intervals is None:
        survival_intervals = [[0.267, 0.445]]
    alpha_n = 0.028 
    alpha_m = 0.147 

    config = {}

    id_kernel_config = {} 
    id_kernel_config["name"] = "SmoothLifeKernel"
    id_kernel_config["kernel_kwargs"] = {}
    id_kernel_config["kernel_kwargs"]["r_a"] = 1.0 / 3.0
    id_kernel_config["kernel_kwargs"]["r_i"] = 0.0
    id_kernel_config["radius"] = radius

    kernel_config = {}
    kernel_config["name"] = "SmoothLifeKernel"
    kernel_config["kernel_kwargs"] = {}
    kernel_config["kernel_kwargs"]["r_a"] = 1.05
    kernel_config["kernel_kwargs"]["r_i"] = 1.0 / 3.0

    kernel_config["radius"] = radius

    gen_config = {}
    gen_config["name"] = "SmoothIntervals"
    gen_config["parameters"] = torch.tensor(birth_intervals)
    gen_config["alpha"] = torch.tensor(alpha_n)
    gen_config["mode"] = 1

    per_config = {}
    per_config["name"] = "SmoothIntervals"
    per_config["parameters"] = torch.tensor(survival_intervals)
    per_config["alpha"] = torch.tensor(alpha_m)
    per_config["mode"] = 1

    config["identity_kernel_config"] = id_kernel_config
    config["neighborhood_kernel_config"] = kernel_config 
    config["genesis_config"] = gen_config
    config["persistence_config"] = per_config

    config["dt"] = 1.0

    return config

def get_life_like_config(radius=1, birth=[3], survival=[2,3]):
    """
    Return a glaberish config based on a Life-like CA's B/S rules.
    The default radius of 1 corresponds to a 3x3 Moore neighborhood. 
    """

    birth_intervals = get_smooth_interval(birth)
    survival_intervals = get_smooth_interval(survival)

    config = {}

    kernel_config = {}
    kernel_config["name"] = "MooreLike"
    kernel_config["radius"] = radius

    gen_config = {}
    gen_config["name"] = "SmoothIntervals"
    gen_config["parameters"] = torch.tensor(birth_intervals)
    gen_config["mode"] = 1

    per_config = {}
    per_config["name"] = "SmoothIntervals"
    per_config["parameters"] = torch.tensor(survival_intervals)
    per_config["mode"] = 1

    config["neighborhood_kernel_config"] = kernel_config 
    config["genesis_config"] = gen_config
    config["persistence_config"] = per_config

    config["dt"] = 1.0

    return config

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


