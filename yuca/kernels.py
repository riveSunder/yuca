import numpy as np

import torch

from yuca.activations import Gaussian, DoGaussian, \
        CosOverX2, GaussianMixture, SmoothLifeKernel

import matplotlib.pyplot as plt

def get_kernel(kernel_config):

    if kernel_config["name"] == "Gaussian":
        get_kernel_fn = Gaussian
    elif kernel_config["name"] == "GaussianMixture":
        get_kernel_fn = GaussianMixture
    elif kernel_config["name"] == "CosOverX2":
        get_kernel_fn = CosOverX2
    elif kernel_config["name"] == "DoGaussian":
        get_kernel_fn = DoGaussian
    elif kernel_config["name"] == "SmoothLifeKernel":
        get_kernel_fn = SmoothLifeKernel
    elif kernel_config["name"] == "InnerMoore":
        kernel = np.zeros((3,3))
        kernel[1,1] = 1.0
        return kernel
    elif kernel_config["name"] == "MooreLike":
        kernel_radius = kernel_config["radius"]
        mid_kernel = kernel_radius 

        kernel = np.ones((kernel_radius * 2 + 1, kernel_radius * 2 + 1))
        kernel[mid_kernel, mid_kernel] = 0.0
        kernel /= kernel.sum()

        return kernel
    else:
        kernel_error = f"kernel function {kernel_config['name']} not found"
        assert False, kernel_error

    
    kernel_fn = get_kernel_fn(**kernel_config["kernel_kwargs"])
    radius = kernel_config["radius"]

    eps = 1e-9
    xx, yy = np.meshgrid(np.arange(-radius, radius + 1), \
            np.arange(-radius, radius + 1))
    grid = np.sqrt(xx**2 + yy**2) / kernel_config["radius"] 

    kernel = kernel_fn(grid.reshape(1, 1, radius * 2 + 1, radius * 2 + 1))
    kernel = kernel - kernel.min()
    kernel = kernel / (eps + kernel.sum())

    return kernel
   

def get_laplacian_kernel(radius=1):

    return  torch.tensor([[[[0,1.,0],[1.,-4.,1.],[0,1.,0]]]])

def get_cosx2_kernel(radius=13, mu=0.5, omega=12.56, r_scale=1.0):
    
    eps = 1e-9

    xx, yy = np.meshgrid(np.arange(-radius, radius + 1), \
            np.arange(-radius, radius + 1))

    grid = np.sqrt(xx**2 + yy**2) / radius * r_scale

    my_fn  = CosOverX2(mu=mu, omega=omega)
    
    kernel = my_fn(grid.reshape(1, 1, radius * 2 + 1, radius * 2 + 1))
    kernel = kernel - kernel.min()
    kernel = kernel / (eps + kernel.sum())

    return kernel

def get_gaussian_kernel(radius=13, mu=0.5, sigma=0.15, r_scale=1.0):
    
    eps = 1e-9

    xx, yy = np.meshgrid(np.arange(-radius, radius + 1), \
            np.arange(-radius, radius + 1))

    grid = np.sqrt(xx**2 + yy**2) / radius * r_scale

    gaussian = Gaussian(mu=mu, sigma=sigma)
    
    kernel = gaussian(grid.reshape(1, 1, radius * 2 + 1, radius * 2 + 1))
    kernel = kernel - kernel.min()
    kernel = kernel / (eps + kernel.sum())

    return kernel

def get_gaussian_mixture_kernel(radius=13, parameters=[0.5, 0.15], r_scale=1.0):
    eps = 1e-9

    xx, yy = np.meshgrid(np.arange(-radius, radius + 1), \
            np.arange(-radius, radius + 1))

    grid = np.sqrt(xx**2 + yy**2) / radius * r_scale

    gaussian = GaussianMixture(parameters=parameters)
    kernel = gaussian(grid.reshape(1, 1, radius * 2 + 1, radius * 2 + 1))
    kernel = kernel - kernel.min()
    kernel = kernel / (eps + kernel.sum())

    return kernel

def get_dogaussian_kernel(radius=13, mu=0.5, sigma=0.15):

    eps = 1e-9
    xx, yy = np.meshgrid(np.arange(-radius, radius + 1), \
            np.arange(-radius, radius + 1))

    grid = np.sqrt(xx**2 + yy**2) / radius

    dogaussian = DoGaussian(mu=mu, sigma=sigma)
    
    kernel = dogaussian(grid.reshape(1, 1, radius * 2 + 1, radius * 2 + 1))
    kernel = kernel - kernel.min()
    kernel = kernel / (eps + kernel.sum())

    return kernel

def get_gaussian_edge_kernel(radius=13, mu=0.5, sigma=0.15, mode=0, dx=0.01):

    eps = 1e-9
    xx, yy = np.meshgrid(np.arange(-radius, radius + 1), \
            np.arange(-radius, radius + 1))
    
    if mode:
        grid1 = np.sqrt((xx - dx)**2 + yy**2) / radius
        grid2 = np.sqrt((xx + dx)**2 + yy**2) / radius
    else:
        grid1 = np.sqrt(xx**2 + (yy - dx)**2) / radius
        grid2 = np.sqrt(xx**2 + (yy + dx)**2) / radius

    gaussian = Gaussian(mu=mu, sigma=sigma)
    
    kernel1 = gaussian(grid1.reshape(1, 1, radius * 2 + 1, radius * 2 + 1))
    kernel2 = gaussian(grid2.reshape(1, 1, radius * 2 + 1, radius * 2 + 1))

    kernel = kernel1 - kernel2
    kernel = kernel - kernel.min()
    kernel = kernel / (eps + kernel.sum())

    return kernel


def get_dogaussian_edge_kernel(radius=13, mu=0.5, sigma=0.15, mode=0, dx=0.01):

    eps = 1e-9
    xx, yy = np.meshgrid(np.arange(-radius, radius + 1), \
            np.arange(-radius, radius + 1))
    
    if mode:
        grid1 = np.sqrt((xx - dx)**2 + yy**2) / radius
        grid2 = np.sqrt((xx + dx)**2 + yy**2) / radius
    else:
        grid1 = np.sqrt(xx**2 + (yy - dx)**2) / radius
        grid2 = np.sqrt(xx**2 + (yy + dx)**2) / radius

    dogaussian = DoGaussian(mu=mu, sigma=sigma)
    
    kernel1 = dogaussian(grid1.reshape(1, 1, radius * 2 + 1, radius * 2 + 1))
    kernel2 = dogaussian(grid2.reshape(1, 1, radius * 2 + 1, radius * 2 + 1))

    kernel = kernel1 - kernel2
    kernel = kernel - kernel.min()
    kernel = kernel / (eps + kernel.sum())

    return kernel

if __name__ == "__main__":

    pass
