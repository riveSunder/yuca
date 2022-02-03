import argparse
import os
import sys
import subprocess

import time

import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

import yuca
from yuca.multiverse import CA
from yuca.utils import query_kwargs, get_bite_mask, save_fig_sequence
from yuca.cppn import CPPN

class Phanes():
    
    def __init__(self, **kwargs):

        self.ca = CA(**kwargs)
        self.ca_steps = query_kwargs("ca_steps", 2048, **kwargs)
        self.batch_size = query_kwargs("batch_size", 128, **kwargs)

        self.dim = query_kwargs("dim", 128, **kwargs)
        self.my_device = query_kwargs("device", "cpu", **kwargs)
        self.use_cppn = query_kwargs("use_cppn", False, **kwargs)

        self.input_directory = query_kwargs("input_directory", None, **kwargs)

        if self.input_directory is None:
            self.input_directory = "configs"

        self.log = {"kwargs": kwargs}

    def load_ca_config(self, filepath):
        pass

    def autocorrelate(self, grid1, grid2):
        
        eps = 1e-7

        grid1 = grid1.detach().cpu().numpy()
        grid2 = grid2.detach().cpu().numpy()
        pad_dim = np.max(np.array(grid1.shape[-2:]) * 2)

        f_g_1 = np.fft.fft2(grid1, (pad_dim, pad_dim))
        f_g_2 = np.fft.fft2(grid2, (pad_dim, pad_dim))

        f_corr_12 = f_g_1 * f_g_2.conj()
        #f_corr_11 = f_g_1 * f_g_1.conj()

        #baseline = np.fft.ifftshift(np.fft.ifft2(f_corr_11)).real
        corr_12 = np.fft.ifftshift(np.fft.ifft2(f_corr_12)).real

        #corr = (eps + corr_12) / (eps + baseline)
        corr = corr_12

        return corr

    def analyze_universe(self):
        
        steps = []
        autocorrelation = []
        mortality_ratio = []
        fertility_ratio = []

        # keep track of how long before patterns vanish
        last_step = - torch.ones(self.batch_size)
        
        grid = torch.zeros(self.batch_size, 1, self.dim, self.dim)
        # b_dim stands for bounded dim
        b_dim = self.dim // 2
        if self.use_cppn:
            for cppn_num in range(self.batch_size):
                cppn = CPPN(dim = self.dim)
                grid[cppn_num] = cppn.get_action()
                grid[:,:,:b_dim//2,:] *= 0
                grid[:,:,:,:b_dim//2] *= 0
                grid[:,:,-b_dim//2:,:] *= 0
                grid[:,:,:,-b_dim//2:] *= 0
        else:

            grid[:,:,b_dim//2:-b_dim//2, b_dim//2:-b_dim//2] = torch.rand(\
                    self.batch_size, 1, b_dim, b_dim)

        # escape mask checks for patterns that escape the bounding box
        escape_mask = torch.zeros_like(grid)
        escape_mask[:,:,0,:] = 1.0
        escape_mask[:,:,-1,:] = 1.0
        escape_mask[:,:,:,0] = 1.0
        escape_mask[:,:,:,-1] = 1.0

        # move data and module to device
        last_step = last_step.to(self.my_device)
        escape_mask = escape_mask.to(self.my_device)
        grid = grid.to(self.my_device)
        self.ca.to_device(self.my_device)

        log_every = max([self.ca_steps // 8, 1])
        old_grid = 1.0 * grid

        for step in range(self.ca_steps):
            
            grid = self.ca(grid)

            if grid.mean() == 0.0:
                break

            active_grids = 1.0 * \
                    (grid.mean(dim=(2,3)).mean(dim=-1) > 0.00)

            fertile_grids = 1.0 * \
                    ((escape_mask * grid).sum(dim=(2,3)).sum(dim = -1) > 0.00)

            autocorr = self.autocorrelate(grid, old_grid)
            old_grid = 1.0 * grid

            autocorrelation.append([np.mean(autocorr), np.std(autocorr), \
                    np.min(autocorr), np.max(autocorr)])
            # keep track of last steps
            last_step[(last_step * (active_grids == 0)) == -1] = step
            # mortality is the potential for patterns to vanish
            mortality_ratio.append(1.0 - active_grids.mean().cpu().item()) #.numpy())
            # rules are fertile when a pattern escapes a bounding box
            fertility_ratio.append(fertile_grids.mean().cpu().item()) #.numpy())
            steps.append(step)


        results = {}

        results["step"] = steps
        results["fertility_ratio"] = fertility_ratio
        results["mortality_ratio"] = mortality_ratio
        results["last_step"] = last_step.cpu().numpy()
        results["autocorrelation"] = autocorrelation

        return results


    def analyze_multiverse(self):

        abs_path = os.path.abspath(os.path.curdir)
        my_directory = os.path.join(abs_path, self.input_directory)

        list_dir = os.listdir(my_directory)

        for filename in list_dir:
            if "npy" in filename:
                load_path = os.path.join(my_directory, filename)

                ca_config = np.load(load_path, allow_pickle=True).reshape(1)[0]
                self.ca.load_config(ca_config)
                self.ca.no_grad()

                t0 = time.time()
                results = self.analyze_universe()
                t1 = time.time()

                elapsed_time = t1-t0

                progress_msg = f"ca config {filename} analyzed in {elapsed_time} s\n"
                progress_msg += f"   avg./final mortality ratio: "
                progress_msg += f"{np.mean(results['mortality_ratio'])}/"
                progress_msg += f"{results['mortality_ratio'][-1]}\n"
                progress_msg += f"   avg./final fertility ratio: "
                progress_msg += f"{np.mean(results['fertility_ratio'])}/"
                progress_msg += f"{results['fertility_ratio'][-1]}\n"
                progress_msg += f"   avg./final autocorrelation: "
                progress_msg += f"{np.mean(results['autocorrelation'])}/"
                progress_msg += f"{results['autocorrelation'][-1][0]}\n"

                print(progress_msg)

                save_directory = os.path.split(os.path.split(my_directory)[0])[0]
                save_directory = os.path.join(save_directory, "ca_analysis")

                
                save_name = os.path.splitext(filename)[0] \
                        + "{self.use_cppn}_analysis.npy"
                save_path = os.path.join(save_directory, save_name)

                results["name"] = filename
                results["use_cppn"] = self.use_cppn
                results["max_ca_steps"] = self.ca_steps
                np.save(save_path, results)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("args for plotting evo logs")

    parser.add_argument("-i", "--input_directory", type=str,\
            default="ca_configs/", help="directory containing ca rule config files")

    parser.add_argument("-d", "--device", type=str, \
            default="cpu", help="device to use (cpu or cuda)")

    parser.add_argument("-c", "--ca_steps", type=int, \
            default=1024, help="number of ca steps to search for")

    parser.add_argument("-u", "--use_cppn", type=int, \
            default=0, help="use CPPNs instead of random initializations")

    args = parser.parse_args()

    kwargs = dict(args._get_kwargs())

    # use subprocess to get the current git hash, store
    hash_command = ["git", "rev-parse", "--verify", "HEAD"]
    git_hash = subprocess.check_output(hash_command)
    # check_output returns bytes, convert to utf8 encoding before storing
    kwargs["git_hash"] = git_hash.decode("utf8")

    # store the command-line call for this experiment
    entry_point = []
    entry_point.append(os.path.split(sys.argv[0])[1])
    args_list = sys.argv[1:]

    sorted_args = []
    for aa in range(0, len(args_list)):

        if "-" in args_list[aa]:
            sorted_args.append([args_list[aa]])
        else: 
            sorted_args[-1].append(args_list[aa])

    sorted_args.sort()
    entry_point.extend(sorted_args)
    kwargs["entry_point"] = entry_point

    phanes = Phanes(**kwargs)

    phanes.analyze_multiverse()

