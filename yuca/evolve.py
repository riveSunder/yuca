import sys
import os
import argparse
import subprocess

import torch

from yuca.cmaes import CMAES, CMACES
from yuca.cppn import CPPN, CPPNPlus

from yuca.wrappers.halting_wrapper import SimpleHaltingWrapper, HaltingWrapper
from yuca.wrappers.random_wrapper import RandomWrapper
from yuca.wrappers.glider_wrapper import GliderWrapper

from yuca.ca.continuous import CCA
from yuca.ca.reaction_diffusion import RxnDfn
from yuca.ca.neural import NCA

from yuca.metaca import MetaCA
from yuca.code import CODE
from yuca.lenia import Lenia
from yuca.random_step_ca import RandomStepCA


def coevolve(**kwargs):

    kwargs["agent_fn"] = CPPNPlus
    kwargs["env_fn"] = GliderWrapper 
    
    if "MetaCA" in kwargs["ca_fn"]:
        kwargs["ca_fn"] = MetaCA
    elif "CCA" in kwargs["ca_fn"]:
        kwargs["ca_fn"] = CCA
    elif "NCA" in kwargs["ca_fn"]:
        kwargs["ca_fn"] = NCA
    elif "RxnDfn" in kwargs["ca_fn"]:
        kwargs["ca_fn"] = RxnDfn
    elif "CODE" in kwargs["ca_fn"]:
        kwargs["ca_fn"] = CODE
    elif "Lenia" in kwargs["ca_fn"]:
        kwargs["ca_fn"] = Lenia
    elif "RandomStepCA" in kwargs["ca_fn"]:
        kwargs["ca_fn"] = RandomStepCA
    else:
        exception_msg = f"should be unreachable, env_fn {kwargs['ca_fn']} "\
                f" not recognized"
        assert False, exception_msg

    print(f"Co-evolve patterns and rules with {kwargs['env_fn']}i reward"\
            f" and CA type {kwargs['ca_fn']}")

    population = CMACES(**kwargs)

    population.search()

def pattern_search(**kwargs):
    
    kwargs["agent_fn"] = CPPN
    kwargs["env_fn"] = GliderWrapper 

    if "MetaCA" in kwargs["ca_fn"]:
        kwargs["ca_fn"] = MetaCA
    elif "CCA" in kwargs["ca_fn"]:
        kwargs["ca_fn"] = CCA
    elif "NCA" in kwargs["ca_fn"]:
        kwargs["ca_fn"] = NCA
    elif "RxnDfn" in kwargs["ca_fn"]:
        kwargs["ca_fn"] = RxnDfn
    elif "CODE" in kwargs["ca_fn"]:
        kwargs["ca_fn"] = CODE
    elif "Lenia" in kwargs["ca_fn"]:
        kwargs["ca_fn"] = Lenia
    elif "RandomStepCA" in kwargs["ca_fn"]:
        kwargs["ca_fn"] = RandomStepCA
    else:
        exception_msg = f"should be unreachable, env_fn {kwargs['ca_fn']} "\
                f" not recognized"
        assert False, exception_msg

    print(f"Evolve mobile patterns with CPPNs and {kwargs['env_fn']}"\
            f" and CA type {kwargs['ca_fn']}")

    population = CMAES(**kwargs)

    population.search()

def universe_search(**kwargs):

    if "SimpleHaltingWrapper" in kwargs["env_fn"]:
        env_fn = SimpleHaltingWrapper
    elif "HaltingWrapper" in kwargs["env_fn"]:
        env_fn = HaltingWrapper
    elif "RandomWrapper" in kwargs["env_fn"]:
        env_fn = RandomWrapper
    else:
        exception_msg = f"should be unreachable, env_fn {kwargs['env_fn']} "\
                f" not recognized"
        assert False, exception_msg

    if "MetaCA" in kwargs["ca_fn"]:
        kwargs["ca_fn"] = MetaCA
    elif "CCA" in kwargs["ca_fn"]:
        kwargs["ca_fn"] = CCA
    elif "NCA" in kwargs["ca_fn"]:
        kwargs["ca_fn"] = NCA
    elif "RxnDfn" in kwargs["ca_fn"]:
        kwargs["ca_fn"] = RxnDfn
    elif "CODE" in kwargs["ca_fn"]:
        kwargs["ca_fn"] = CODE
    elif "Lenia" in kwargs["ca_fn"]:
        kwargs["ca_fn"] = Lenia
    elif "RandomStepCA" in kwargs["ca_fn"]:
        kwargs["ca_fn"] = RandomStepCA
    else:
        exception_msg = f"should be unreachable, env_fn {kwargs['ca_fn']} "\
                f" not recognized"
        assert False, exception_msg

    print(f"Evolve CA rules with {kwargs['env_fn']} "\
            f" and CA type {kwargs['ca_fn']}")

    kwargs["env_fn"] = env_fn

    population = CMAES(**kwargs)

    population.search()
    
def evolve(**kwargs):

    if "float64" in kwargs["dtype"]:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    if "pattern" in kwargs["tag"]:
        pattern_search(**kwargs)
    elif "coevo" in kwargs["tag"]:
        coevolve(**kwargs)
    else:
        universe_search(**kwargs)

    print("evolution completed successfully")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("args for plotting evo logs")

    parser.add_argument("-b", "--batch_size", type=int, \
            default=64, help="number of grid instances (vectorization)")
    parser.add_argument("-c", "--ca_steps", type=int, \
            default=1024, help="number of ca steps to search for")
    parser.add_argument("-ca", "--ca_fn", type=str, default="CCA")

    parser.add_argument("-cc", "--ca_config", type=str, default=None,\
            help="filename (or filepath) designating a ca_config to load")
    parser.add_argument("-ch", "--internal_channels", type=int, default=1,\
            help="number of internal channels in ca system")

    parser.add_argument("-d", "--device", type=str, \
            default="cpu", help="device to use (cpu or cuda)")

    parser.add_argument("-dt", "--dt", type=float, \
            default=0.3, help="step size used by model")

    parser.add_argument("-dtype", "--dtype", type=str, \
            default="float32", \
            help="set default dtype in torch")

    parser.add_argument("-e", "--selection_mode", type=int, default=0, \
            help="selection mode: 0: truncation, 1: rand. tourney, 2: proportional")

    parser.add_argument("-f", "--env_fn", type=str, default="HaltingWrapper")
    parser.add_argument("-g", "--generations", type=int, default=32, \
            help="number of generations to train for")

    parser.add_argument("-i", "--input_filepath", type=str, \
            default=None, \
            help="npy log file training curves etc.")

    parser.add_argument("-k", "--kernel_radius", type=int, \
            default=13, help="kernel radius. kernel shape will be 2r+1 by 2r+1)")
    parser.add_argument("-kp", "--kernel_peaks", type=int, \
            default=3, help="number of gaussian peaks for the neighborhood kernel(s)")
    parser.add_argument("-l", "--replicates", type=int, default=1,\
            help="number of replicates to use in get_fitness")
    parser.add_argument("-m", "--dim", type=int, \
            default=128, help="grid x,y dimension (square edge length)")

    parser.add_argument("-p", "--population_size", type=int, default=32, \
            help="number of individuals in population")
    parser.add_argument("-r", "--prediction_mode", type=int, default=0, \
            help="prediction mode: 0-vanishing, 1-static end, 2-both")

    parser.add_argument("-s", "--seed", type=int, nargs="+", default=13,\
            help="seeds to initialized PRNGs, can enter more than one")
    parser.add_argument("-t", "--tag", type=str, \
            default="pattern_search", \
            help="string tag for identifying experiments")
    parser.add_argument("-v", "--conv_mode", type=str, \
            default="circular", \
            help="padding mode to use, 'circular', 'reflect', or 'zeros'")
    parser.add_argument("-w", "--workers", type=int, default=0,\
            help="number of thread to use as workers, 0 - no workers (mantle only)")


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

    evolve(**kwargs)

