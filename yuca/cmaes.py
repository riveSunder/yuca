import sys
import argparse
import subprocess
import os

from random import shuffle
import time
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

from yuca.utils import query_kwargs, seed_all, save_fig_sequence

from yuca.params_agent import ParamsAgent 
from yuca.wrappers.halting_wrapper import SimpleHaltingWrapper, HaltingWrapper 
from yuca.wrappers.glider_wrapper import GliderWrapper

import matplotlib.pyplot as plt

from mpi4py import MPI
comm = MPI.COMM_WORLD

class CMAES():

    def __init__(self, **kwargs):

        self.workers = query_kwargs("workers", 0, **kwargs)
        self.tag = query_kwargs("tag", "", **kwargs)
        self.my_seed = query_kwargs("seed", [42], **kwargs)

        if type(self.my_seed) is not list:
            self.my_seed = [self.my_seed]

        self.dim = query_kwargs("dim", 64, **kwargs)
        self.ca_steps = query_kwargs("ca_steps", 512, **kwargs)
        self.replicates = query_kwargs("replicates", 3, **kwargs)
        self.batch_size = query_kwargs("batch_size", 8, **kwargs)
        self.population_size = query_kwargs("population_size", 16, **kwargs)        
        self.elite_keep = self.population_size // 8 + 1
        self.generations = query_kwargs("generations", 16, **kwargs)
        self.selection_mode = query_kwargs("selection_mode", 0, **kwargs)
        self.elitism = 1

        self.kernel_radius = query_kwargs("kernel_radius", 13, **kwargs)
        self.prediction_mode = query_kwargs("prediction_mode", 0, **kwargs)

        env_fn = query_kwargs("env_fn", HaltingWrapper, **kwargs)
        
        self.env = env_fn(lr = 1e-2, dropout = 0.125, **kwargs) 

        self.workers = query_kwargs("workers", 0, **kwargs)
        self.my_device = query_kwargs("device", "cpu", **kwargs)

        self.agent_fn = query_kwargs("agent_fn", ParamsAgent, **kwargs)

        self.kwargs = kwargs

        # the type of evolution (pattern or universe rule selection)
        # is determined by the presence or absence of the string
        # 'pattern' in the tag.

        if "pattern" in self.tag:
            temp_agent = self.agent_fn(**kwargs)
            self.starting_means = temp_agent.get_params()
            covar_weights = np.ones_like(self.starting_means)*1.00
            self.external_channels = self.env.ca.external_channels
        else:
            ca_params = self.env.ca.get_params()
            self.external_channels = self.env.ca.external_channels
            kwargs["external_channels"] = self.external_channels
            temp_agent = self.agent_fn(ca_params = ca_params,  **kwargs)
            self.starting_means = temp_agent.get_params()
            covar_weights = np.ones_like(self.starting_means)*0.15
            covar_weights[1:covar_weights.shape[0]] *= 5e-4 
            

        self.starting_covar = np.abs(np.diag(covar_weights))

        print(f"number of params per agent = {self.starting_means.shape}")

        self.exp_id = f"exp_{self.tag}_{int(time.time())}"

        self.input_filepath = query_kwargs("input_filepath", None, **kwargs)
        print(self.input_filepath)

        self.elite_params = None

        if self.input_filepath is not None:
            my_data = np.load(self.input_filepath, allow_pickle=True).reshape(1)[0]

            my_params = my_data["elite_params"]
            if (my_params[-1][0].shape[0] == self.starting_means.shape[0]):

                for ii in range(self.elite_keep):

                    if self.elite_params is None:
                        self.elite_params = my_params[-1][ii:ii+1]
                    else:
                        self.elite_params = np.append(self.elite_params,\
                                my_params[-1][ii:ii+1], axis=0)

                self.starting_means = np.mean(self.elite_params, axis=0)

            else:
                params_msg = f"log and agent params have different shape"\
                        f" of {my_params[-1][0].shape} and {self.starting_means.shape}"\
                        f" \n    using default distribution initialization for "\
                        f" agent_fn {self.agent_fn.__class__.__name__}"

                print(params_msg)
                

    def sample_distribution(self):

        return np.random.multivariate_normal(self.means, self.covar)
        
    def reset(self):

        self.means = 1.0 * self.starting_means
        self.covar = 1.0 * self.starting_covar
        self.population = []
        self.initialize_population()

    def initialize_population(self):
        
        if self.elite_params is not None and self.elitism:
            for hh in range(self.elite_params.shape[0]):
            
                self.population.append(self.agent_fn(\
                        params = self.elite_params[hh].squeeze(),\
                        external_channels = self.external_channels,\
                        dim = self.dim))

                self.population[hh].to_device(self.my_device)

            start_population = self.elite_params.shape[0]
        else:
            start_population = 0

        for ii in range(start_population, self.population_size):

            self.population.append(self.agent_fn(\
                    params = self.sample_distribution(),\
                    external_channels = self.external_channels,\
                    dim = self.dim))

            self.population[ii].to_device(self.my_device)


        
    def get_fitness(self, agent_index, steps=10, replicates=1, seed=13):

        fitness_replicates = []
        seed_all(seed)
        proportion_alive = 0.0

        for replicate in range(replicates):

            fitness = 0.0

            self.env.reset()
            self.env.to_device(self.my_device)
            self.env.eval()

            for step in range(steps):
                action = self.population[agent_index].get_action()
                o, r, d, info = self.env.step(action)

                fitness += r
                proportion_alive += info["active_grid"].detach().cpu()

            fitness /= steps
            fitness_replicates.append(fitness.detach().cpu().numpy())

        proportion_alive /= (steps * replicates)

        # return the worst performing replicate,
        # reduce the impact of lucky/unlucky predictor initializations
        return np.mean(fitness_replicates), proportion_alive

    def rank_population(self, fitness, seed=0):
        """
        for the final generation, fitness scores are ranked deterministically
        """

        print("final population sorting, truncation mode")

        sorted_indices = list(np.argsort(fitness))
        sorted_indices.reverse()
        sorted_fitness = np.array(fitness)[sorted_indices]

        elite_pop = []
        elite_fitness = []

        for jj in range(self.elite_keep):

            elite_pop.append(self.population[sorted_indices[jj]])
            elite_fitness.append(fitness[sorted_indices[jj]])

        elite_params = None
        for jj in range(self.elite_keep):

            if elite_params is None:
                elite_params = elite_pop[jj].get_params()[np.newaxis,:]
            else:
                elite_params = np.append(elite_params,\
                        elite_pop[jj].get_params()[np.newaxis,:],\
                        axis=0)

        self.elite_params = elite_params
        elite_mean = np.mean(elite_params, axis=0)

        covar = (1 / self.elite_keep) \
                * np.matmul((elite_params - self.means).T,\
                (elite_params - self.means))

        self.means = elite_mean
        self.covar = covar

        # save best config


        this_filepath = os.path.realpath(__file__)
        temp = os.path.split(os.path.split(this_filepath)[0])[0]
        ca_config_filepath = os.path.join(temp, f"ca_configs/{self.exp_id}_seed{seed}.npy")
        
        if "pattern" not in self.tag:
            self.env.ca.set_params(self.elite_params[0])

        self.env.ca.save_config(ca_config_filepath)


    def update_population(self, fitness):

        if self.selection_mode == 0:
            # selection mode 0, truncation selection

            sorted_indices = list(np.argsort(fitness))
            sorted_indices.reverse()
            sorted_fitness = np.array(fitness)[sorted_indices]

            elite_pop = []
            elite_fitness = []

            for jj in range(self.elite_keep):

                elite_pop.append(self.population[sorted_indices[jj]])
                elite_fitness.append(fitness[sorted_indices[jj]])

        elif self.selection_mode == 1:
            # selection mode 1, tournament selection
            # randomly make a bracket (w/ replacement) 
            # then pick the champion from that bracket
            # repeat to fill elite population

            elite_pop = []
            elite_fitness = []
            for elite_member in range(self.elite_keep):

                indices = np.random.randint(0, self.population_size, \
                        size = (self.elite_keep,))

                temp_fitness = np.array(fitness)[indices]
                champion_index = np.argmax(temp_fitness)

                elite_pop.append(self.population[indices[champion_index]])
                elite_fitness.append(fitness[indices[champion_index]])

        elif self.selection_mode == 2:
            # fitness proportional selection


            sorted_indices = list(np.argsort(fitness))
            sorted_indices.reverse()
            sorted_fitness = np.array(fitness)[sorted_indices]

            probability = np.array([1/elem**2 \
                    for elem in range(1,len(sorted_fitness) + 1)])
            probability = probability / probability.sum()

            elite_pop = []
            elite_fitness = []

            lottery_indices = np.random.choice(sorted_indices, p=probability, \
                    size = (self.elite_keep,), replace=False)

            for index in lottery_indices:

                elite_pop.append(self.population[index])
                elite_fitness.append(fitness[index])

        elite_params = None
        for jj in range(self.elite_keep):

            if elite_params is None:
                elite_params = elite_pop[jj].get_params()[np.newaxis,:]
            else:
                elite_params = np.append(elite_params,\
                        elite_pop[jj].get_params()[np.newaxis,:],\
                        axis=0)

        self.elite_params = elite_params
        elite_mean = np.mean(elite_params, axis=0)

        covar = (1 / self.elite_keep) \
                * np.matmul((elite_params - self.means).T,\
                (elite_params - self.means))

        self.means = elite_mean
        self.covar = covar

        
        for kk in range(self.elite_keep * self.elitism // 2):
            self.population[kk].set_params(elite_pop[kk].get_params())

        for ll in range(self.elite_keep * self.elitism // 2, self.population_size):
            self.population[ll].set_params(self.sample_distribution())    

        # shuffle pop, this will be important for tourney selection later
        #shuffle(self.population)

    def save_gif(self, tag="", starting_grid=None):


        if starting_grid is None:
            starting_grid = torch.rand(1, self.env.ca.external_channels, \
                    self.dim, self.dim)
            half = self.dim // 4
            starting_grid[:,:, :half, :] *= 0
            starting_grid[:,:, :, :half] *= 0
            starting_grid[:,:, -half:, :] *= 0
            starting_grid[:,:, :, -half:] *= 0

        #self.env.ca.set_params(self.population[0].get_params())
        grid = self.env.ca(starting_grid.to(self.env.ca.my_device)).to("cpu")
        restore_device = torch.device(self.env.ca.my_device).type
        restore_index = torch.device(self.env.ca.my_device).index

        if restore_index is not None:
                restore_device += f":{restore_index}"

        self.env.ca.to_device("cpu")

        save_fig_sequence(grid, self.env.ca, num_steps=256, mode=0, \
                frames_path=f"./assets/{self.exp_id}",\
                tag=tag, cmap=plt.get_cmap("magma"))

        # send ca model back to device
        self.env.ca.to_device(restore_device)

    def get_elite_configs(self):

        elite_configs = []
        
        restore_steps = 1 * self.env.ca_steps
        self.env.ca_steps = 2
        for hh in range(self.elite_params.shape[0]):

            
            temp_agent = self.agent_fn(\
                    params = self.elite_params[hh].squeeze(),\
                    external_channels = self.external_channels,\
                    dim = self.dim)

            temp_agent.to_device(self.my_device)

            action = temp_agent.get_action()
            _ = self.env.step(action)

            elite_configs.append(self.env.ca.make_config())

        self.env.ca_steps = restore_steps
        return elite_configs

    def mpi_fork(self):
        """
        relaunches the current script with workers
        Returns "parent" for original parent, "child" for MPI children
        (from https://github.com/garymcintire/mpi_util/)
        via https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease
        """
        global num_worker, rank

        if self.workers <= 1:
            print("if n<=1")
            num_worker = 0
            rank = 0
            return "child"

        if os.getenv("IN_MPI") is None:
            env = os.environ.copy()
            env.update(\
                    MKL_NUM_THREADS="1", \
                    OMP_NUM_THREAdS="1",\
                    IN_MPI="1",\
                    )
            print( ["mpirun", "-np", str(self.workers), sys.executable] + sys.argv)
            subprocess.check_call(["mpirun", "-np", str(self.workers), sys.executable] \
            +['-u']+ sys.argv, env=env)

            return "parent"
        else:
            num_worker = comm.Get_size()
            rank = comm.Get_rank()
            return "child"

    def search(self):

        if self.mpi_fork() == "parent":
            os._exit(0)

        if rank == 0:
            self.mantle()
        else:
            self.arm()

    def mantle(self):

        t0 = time.time()

        progress = {}
        progress["elite_params"] = []
        progress["elite_configs"] = []
        progress["mean_fitness"] = []
        progress["max_fitness"] = []
        progress["min_fitness"] = []
        progress["std_dev_fitness"] = []
        progress["kwargs"] = self.kwargs
        progress["distribution"] = []

        print(f"begin evolution with:\n" \
                f"selection mode {self.selection_mode} " \
                f"prediction motivator mode {self.prediction_mode}")

        self.env.ca.no_grad()
        self.env.ca.to_device(self.my_device)
        for my_seed in self.my_seed:
            print(f"random seed {my_seed}")

            seed_all(my_seed)
            self.reset()
            self.initialize_population()
            
            self.generation = 0

            if "pattern" not in self.exp_id:
                for idx in range(min(self.elite_keep, 4)):
                    self.env.ca.set_params(self.population[idx].get_params())
                    tag = f"{self.exp_id}_seed{my_seed}_"\
                            f"immode{self.prediction_mode}" 
                    tag = [tag, f"gen{0}"]
                    self.save_gif(tag=tag) 
            else:
                for idx in range(min(self.elite_keep,4)):
                    grid = self.population[idx].get_action()
                    effective_steps = min([self.ca_steps, 512])
                    tag = f"{self.exp_id}_seed{my_seed}_"\
                            f"immode{self.prediction_mode}" 
                    tag = [tag, f"gen{0}"]
                    self.save_gif(tag=tag, starting_grid=grid)

            for generation in range(self.generations):
                self.generation = generation
                fitness = []
                proportion_alive = []
                
                t1 = time.time()

                if self.workers == 0:
                    for jj in range(self.population_size):
                        result = self.get_fitness(jj, steps = 1, \
                                replicates = self.replicates, \
                                seed = generation * my_seed)


                        fitness.append(result[0])
                        proportion_alive.append(result[1])

                else:
                    subpopulation_size = int(self.population_size / (self.workers-1))
                    population_remainder = self.population_size % (num_worker-1)
                    population_left = self.population_size

                    batch_end = 0
                    extras = 0

                    # send parameters to arms
                    for cc in range(1, self.workers):
                        run_batch_size = min(subpopulation_size, population_left)

                        if population_remainder:
                            run_batch_size += 1
                            population_remainder -= 1
                            extras += 1

                        batch_start = batch_end 
                        batch_end = batch_start + run_batch_size 

                        parameters_list = [my_agent.get_params() \
                                for my_agent in self.population]

                        agent_indices = [elem for elem in range(batch_start, batch_end)]

                        comm.send((parameters_list, agent_indices, \
                                generation * my_seed), dest=cc)

                    # receive current generation's fitnesses from arm processes
                    population_left = self.population_size
                    for dd in range(1, num_worker):
                        #fit, total_steps, agent_done_at = comm.recv(source=dd)
                        fit, prop_alive = comm.recv(source=dd)

                        fitness.extend(fit)
                        proportion_alive.extend(prop_alive)


                
                if generation == self.generations - 1:
                    self.rank_population(fitness, seed=my_seed)
                else:
                    self.update_population(fitness)

                fit_mean = np.mean(fitness) 
                fit_max = np.max(fitness) 
                fit_min = np.min(fitness) 
                fit_std_dev = np.std(fitness) 
                avg_alive = np.mean(proportion_alive)

                elite_configs = self.get_elite_configs()

                if "pattern" not in self.exp_id:
                    progress["elite_params"].append(self.elite_params)
                    progress["distribution"].append([1.0 * self.means, 1.0 * self.covar])
                else:
                    progress["elite_params"] = [self.elite_params]
                    progress["distribution"] = [[1.0 * self.means, 1.0 * self.covar]]

                progress["elite_configs"].append(elite_configs)

                progress["mean_fitness"].append(fit_mean)
                progress["min_fitness"].append(fit_min)
                progress["max_fitness"].append(fit_max)
                progress["std_dev_fitness"].append(fit_std_dev)

                logs_path = os.path.join("./logs", f"{self.exp_id}_seed{my_seed}.npy")
                np.save(logs_path, progress)

                t2 = time.time()
                timing_msg = f"elapsed time: {t2-t0:.4f}, generation {t2-t1:.4f}"
                proportion_msg = f"avg. proportion of active end grids: {avg_alive}"

                progress_msg = f"gen. {generation}, fitness mean +/- std. dev. " \
                        f"= {fit_mean:.3e} +/- {fit_std_dev:.3e} " \
                        f"max = {fit_max:.3e}, min = {fit_min:.3e}"


                print(timing_msg)
                print(progress_msg)
                print(proportion_msg)


                if fit_std_dev < 0.02 and generation > 6:
                    self.rank_population(fitness, seed = my_seed)
                    early_msg = f"fitness std. dev. fallen to {fit_std_dev} "\
                            f"ending evolution early."
                    print(early_msg)
                    break

            if "pattern" not in self.exp_id:
                for elite_idx in range(self.elite_keep):
                    self.env.ca.set_params(self.population[elite_idx].get_params())
                    tag = f"{self.exp_id}_seed{my_seed}_"\
                            f"immode{self.prediction_mode}" 
                    tag = [tag, f"gen{generation}"]
                    self.save_gif(tag=tag) 
            else:
                for elite_idx in range(self.elite_keep):
                    self.population[elite_idx].set_params(progress["elite_params"][-1][elite_idx])
                    grid = self.population[elite_idx].get_action()
                    effective_steps = min([self.ca_steps, 512])
                    tag = f"{self.exp_id}_seed{my_seed}_"\
                            f"immode{self.prediction_mode}" 
                    tag = [tag, f"gen{generation}_elite{elite_idx}"]
                    self.save_gif(tag=tag, starting_grid=grid)

        for ee in range(1, self.workers):
            print(f"send shutown signal to worker {ee}")
            comm.send((0, 0, 0), dest=ee)


    def arm(self):
        # This will be implemented for parallelization with mpi4py
        while True:
            parameters_list, agent_indices, my_seed = comm.recv(source=0)
            if parameters_list == 0:
                print(f"worker {rank} shutting down")
                break

            self.population_size = len(parameters_list)
            self.reset()

            for ff in range(self.population_size):
                self.population[ff].set_params(parameters_list[ff])

            fitness = []
            prop_alive = []

            for agent_idx in agent_indices:

                result = self.get_fitness(agent_idx, steps=1,\
                        replicates=self.replicates, seed = my_seed)

                fitness.append(result[0])
                prop_alive.append(result[1])

            comm.send((fitness, prop_alive), dest=0)



class CMACES(CMAES):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_fitness(self, agent_index, steps=10, replicates=1, seed=13):

        fitness_replicates = []
        seed_all(seed)
        proportion_alive = 0.0

        for replicate in range(replicates):

            fitness = 0.0

            self.env.reset()
            self.env.to_device(self.my_device)

            for step in range(steps):
                rule_action, pattern_action = self.population[agent_index].get_action()
                
                self.env.ca.set_params(rule_action)

                o, r, d, info = self.env.step(pattern_action)

                fitness += r
                proportion_alive += info["active_grid"].detach().cpu()

            fitness /= steps
            fitness_replicates.append(fitness.detach().cpu().numpy())

        proportion_alive /= (steps * replicates)

        # return the worst performing replicate,
        # reduce the impact of lucky/unlucky predictor initializations
        return np.mean(fitness_replicates), proportion_alive

    def get_elite_configs(self):

        elite_configs = []
        
        restore_steps = 1 * self.env.ca_steps
        self.env.ca_steps = 2
        for hh in range(self.elite_params.shape[0]):

            
            temp_agent = self.agent_fn(\
                    params = self.elite_params[hh].squeeze(),\
                    external_channels = self.external_channels,\
                    dim = self.dim)

            temp_agent.to_device(self.my_device)

            rule_action, pattern_action = temp_agent.get_action()
            _ = self.env.ca.set_params(rule_action)

            elite_configs.append(self.env.ca.make_config())

        self.env.ca_steps = restore_steps
        return elite_configs

    def save_gif(self, tag="", grid=None):

        for elite_idx in range(self.elite_keep):
            if self.elite_params is not None:
                self.population[elite_idx].set_params(self.elite_params[elite_idx])

            rule_action, pattern_action = self.population[elite_idx].get_action()
            self.env.ca.set_params(rule_action)

            effective_steps = min([self.ca_steps, 512])

            #tag = [tag, f"gen{self.generation}"]

            save_fig_sequence(pattern_action, self.env.ca, num_steps=effective_steps,\
                frames_path=f"./assets/{self.exp_id}",\
                tag=tag)#f"{self.exp_id}_{self.my_seed}_elite{elite_idx}")


