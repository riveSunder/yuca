import os
import argparse

import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

from yuca.ca.continuous import CCA
from yuca.utils import query_kwargs, get_bite_mask
from yuca.activations import DoubleGaussian

import matplotlib.pyplot as plt
my_cmap = plt.get_cmap("magma")

def plot_exp(**kwargs):


    input_filepath = query_kwargs("input_filepath", None, **kwargs)

    print(f"input filepath is {input_filepath}")

    if os.path.exists(input_filepath):
        my_data = np.load(input_filepath, allow_pickle=True).reshape(1)[0]
    else:
        assert False, f"path {input_filepath} not found"

    end_params = my_data["elite_params"][-1] 

    for ii, my_params in enumerate(end_params):

        # assuming a double gaussian update function
        genesis_fn = DoubleGaussian(mu = my_params[0:2], sigma=my_params[2:4])
        persistence_fn = DoubleGaussian(mu = my_params[4:6], sigma = my_params[6:8])

        x = np.arange(-0.1, 1.1, 0.0001)

        y_gen = genesis_fn(x).detach().numpy()
        y_per = persistence_fn(x).detach().numpy()

        plt.figure()
        plt.title(f"End params {ii} update rules")

        plt.plot(x, y_gen, "-", lw = 3, alpha = 0.5, color = my_cmap(32), \
                label = "genesis fn")
        plt.plot(x, y_per, "--", lw = 3, alpha = 0.5, color = my_cmap(192), \
                label = "persistence fn")
                
        plt.legend()


    mean_fitness = np.array(my_data["mean_fitness"])
    max_fitness = my_data["max_fitness"]
    min_fitness = my_data["min_fitness"]
    std_dev_fitness = np.array(my_data["std_dev_fitness"])

    plt.figure()
    plt.plot([gen for gen in range(len(mean_fitness))], \
            mean_fitness, lw=3, alpha=0.5, label="mean fitness",\
            color = my_cmap(128))
    plt.plot([gen for gen in range(len(mean_fitness))], \
            max_fitness, "--", lw=1, alpha = 0.95, color = "k",\
            label="mean fitness")
    plt.plot([gen for gen in range(len(mean_fitness))], \
            min_fitness, "--", lw=1, alpha = 0.95, color = "k",\
            label="mean fitness")
    plt.fill_between([gen for gen in range(len(mean_fitness))], \
            mean_fitness - std_dev_fitness, mean_fitness + std_dev_fitness,\
            alpha=0.25, color = my_cmap(128))

    plt.legend()

    plt.show()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("args for plotting evo logs")

    parser.add_argument("-i", "--input_filepath", type=str, \
            default="./logs/exp__1642180973_seed13.npy", \
            help="npy log file training curves etc.")

    args = parser.parse_args()

    kwargs = dict(args._get_kwargs())

    plot_exp(**kwargs)

