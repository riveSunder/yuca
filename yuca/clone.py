import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from yuca.utils import query_kwargs, seed_all
from yuca.ca.neural import NCA
from yuca.ca.continuous import CCA

def clone_from_ca(**kwargs):
    """
    clone a continuous CA rule set specified in the filepath ca_config
    """

    file_directory = os.path.abspath(__file__).split("/")
    root_directory = os.path.join(*file_directory[:-2])
    default_directory = os.path.join("/", root_directory, "ca_configs")
    log_directory = os.path.join("/", root_directory, "logs")


    max_hidden_channels = query_kwargs("max_hidden_channels", 2048, **kwargs)
    error_threshold = query_kwargs("error_threshold", 1e-2, **kwargs)
    max_iterations = query_kwargs("max_iterations", 10, **kwargs)
    max_steps = query_kwargs("max_steps", 1024, **kwargs)
    display_every = max([max_steps // 10, 1])
    batch_size = query_kwargs("batch_size", 32, **kwargs)
    learning_rate = query_kwargs("learning_rate", 1e-3, **kwargs)
    verbose = query_kwargs("verbose", False, **kwargs)
    save_name = query_kwargs("save_name", "default_nca.pt", **kwargs)
    save_path = os.path.join(log_directory, save_name)


    best_state_dict = None
    
    ca_configs = query_kwargs("ca_config", "orbium", **kwargs)
    if type(ca_configs) == str:
        ca_configs = [ca_configs]

    seed_all(42)
    for ca_config in ca_configs:
        hidden_channels = query_kwargs("hidden_channels", 32, **kwargs)

        ca = CCA()
        ca.restore_config(ca_config)
        internal_channels = ca.internal_channels
        external_channels = ca.external_channels

        iteration = 0
        best_error = float("Inf")

        t0 = time.time()
        while iteration < max_iterations:
            iteration_best_error = float("Inf")
            relative_error = float("Inf")

            nca = NCA(internal_channels=internal_channels,\
                    external_channels=external_channels,\
                    hidden_channels=hidden_channels)
            nca.add_neighborhood_kernel(ca.neighborhood_kernels)
            nca.yes_grad()

            optimizer = torch.optim.Adam(nca.weights_layer.parameters(), lr=learning_rate)

            smooth_loss = None
            # loss_alpha used for exponential averaging
            loss_alpha = 0.99
            old_params = nca.get_params()
            for step in range(max_steps):
                
                optimizer.zero_grad()

                x = torch.rand(batch_size, external_channels, 256, 256)
                xn = torch.rand(batch_size, external_channels, 256, 256)
                #x *= (1.0 * torch.rand(batch_size, external_channels, 256, 256) > 0.75)

                target = ca.update_helper(x)
                predicted = nca.update_helper(x)
                #with torch.no_grad():
                #    target = ca.update_universe(xi, xn)
                #target.requires_grad = True

                #predicted = nca.update_universe(xi, xn)
                loss = torch.abs(torch.sqrt((predicted - target)**2)).mean() #F.mse_loss(predicted, target)

                loss.backward()
                optimizer.step()

                if smooth_loss is None:
                    smooth_loss = loss.detach()
                else:
                    smooth_loss = loss.detach() #smooth_loss * loss_alpha + (1-loss_alpha) * loss.detach()

                relative_error = (torch.abs(predicted - target)/(target.abs()+1e-9)).mean()

                if relative_error < iteration_best_error:
                    iteration_best_error = relative_error

                    if relative_error < best_error:
                        best_error = relative_error
                        best_state_dict = nca.state_dict()

                    if verbose:
                        msg = f"iteration {iteration} new best relative error at step {step} "\
                                f"= {iteration_best_error:.3e}, best overall = {best_error:.3e}\n"
                        print(msg)

                    if iteration_best_error < error_threshold:
                        break


                if step % display_every == 0:
                    new_params = nca.get_params()

                    p_diff = (old_params - new_params).mean()
                    t1 = time.time()
                    time_elapsed = t1 - t0
                    msg = f"iteration {iteration} with {hidden_channels} ch_h "\
                            f"step {step} loss; {smooth_loss:.4e}"\
                            f" time elapsed: {time_elapsed:.2f}"\
                            f" param diff mean: {p_diff:.2e}"
                    print(msg)

            hidden_channels = min([max_hidden_channels, int(hidden_channels*1.25)])
            iteration += 1
            

        msg = f"attempted to clone {ca_config} to nca\n"
        msg += f"best nca after {iteration} iterations of max {max_steps} steps \n"

        if best_error < error_threshold:
            msg += f"\tmeets error threshold {error_threshold:.3e} with loss {best_error:.3e}\n"
        else:
            msg += f"\tdid not meet error threshold {error_threshold:.3e} with loss {best_error:.3e}\n"

        if "ca_config" in save_path:
            config_name = os.path.splitext(ca_config)[0] 
            save_path = save_path.replace("ca_config", config_name)

        msg += f"saving to {save_path}"
        print(msg)

        torch.save(best_state_dict, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-cc", "--ca_config", type=str, nargs="+",\
        default=["orbium.npy"],\
        help="CA config filepath (or filename) to clone NCA from. Default: orbium.npy")
    parser.add_argument("-e", "--error_threshold", type=float,\
        default=1e-3,\
        help="mean relative absolute error threshold")
    parser.add_argument("-m", "--max_steps", type=int,\
        default=100,\
        help="maximum batch steps to train/clone")
    parser.add_argument("-i", "--max_iterations", type=int,\
        default=1,\
        help="maximum batch steps to train/clone")
    parser.add_argument("-l", "--learning_rate", type=float,\
        default=1e-3,\
        help="learning rate")
    parser.add_argument("-o", "--save_name", type=str,\
        default="default_nca.pt",\
        help="NCA filepath to save results")
    parser.add_argument("-v", "--verbose", type=int,\
        default=0,\
        help="0- not verbose 1- verbose")


    args = parser.parse_args()

    kwargs = dict(args._get_kwargs())

    clone_from_ca(**kwargs)
