import os

import time

import numpy as np
import scipy
import matplotlib

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42

matplotlib.rcParams["font.size"] = 22
matplotlib.rcParams["ps.fonttype"] = 42

my_cmap = plt.get_cmap("magma")
#matplotlib.rcParams["text.usetex"] = True

import IPython

import skimage
import skimage.io as sio
import skimage.transform

import PIL
from PIL import Image

matplotlib.rcParams["ps.fonttype"] = 42

my_cmap = plt.get_cmap("magma")

# plotting helpers

def plot_kernel_growth(kernel, growth_fn, my_cmap=plt.get_cmap("magma"), \
        titles=["Kernel", "Growth Function"], my_xrange=[-0.1, 1.1],\
        vmin=0, vmax=1, invert=True, cmap_offset=0):
    """
    Plot the neighborhood kernel and continuous growth/update function
    """
    
    # Use `facecolor="white"` because of everyone using dark mode
#    fig, ax = plt.subplots(1, 3, figsize=(3.2,0.8), \
#            gridspec_kw={'width_ratios': [.1, 1, 2.5]}, facecolor="white")
    fig, ax = plt.subplots(1, 3, figsize=(3.0,0.8), \
            gridspec_kw={'width_ratios': [.1, 1, 2.5]}, facecolor="white")

    kernel = kernel / kernel.max()
    if invert:
        display_kernel = 1.0 - my_cmap(np.pad(kernel,((1,1),(1,1))))[:,:,0:3]
    else:
        display_kernel = my_cmap(kernel)[:,:,0:3]

    subplot_a = ax[1].imshow(display_kernel, interpolation="nearest") 
    ax[1].set_title(titles[0], fontsize=8)
    ax[1].set_xticklabels('')
    ax[1].set_yticklabels('')
    
    my_colorbar = (np.arange(1024, 0, -1) / 1023).reshape(-1, 1) * np.ones((1024, 128))

    my_colorbar = my_cmap(my_colorbar)[:,:,:3]
    if invert:
        my_colorbar = 1.0 - my_colorbar
    subplot_b = ax[0].imshow(my_colorbar) #ax[0].imshow(my_colorbar)

    ax[0].set_yticks([1023, 511, 0])
    ax[0].set_yticklabels([0., f"{vmax*0.5:.2e}", f"{vmax:.2e}"], {"fontsize": 6})
    ax[0].set_xticklabels("")
    #ax[0].set_yticklabels("")


    my_line_color = np.array(my_cmap(cmap_offset+96))[:3]
    if invert:
        my_line_color = 1.0 - my_line_color
    x = np.arange(my_xrange[0], my_xrange[1], (my_xrange[1]-my_xrange[0]) / 1000)

    ax_twin = ax[2].twinx()
    subplot_c = ax_twin.plot(x, growth_fn(x), lw=1, color=my_line_color, \
            label="$G_{growth}(\cdot)$")
    ax_twin.set_yticks([-1.0, 0.0, 1.0])
    ax_twin.set_yticklabels([-1.0, 0.0, 1.0], {"fontsize": 6})

    ax[2].set_yticklabels("")

    ax[2].set_xticks([0.0, 1.0])
    ax[2].set_xticklabels([0.0, 1.0], {"fontsize": 6})
    ax[2].set_title(titles[1], fontsize=8)
    
    ax_twin.legend(fontsize=7)    
    plt.tight_layout()
    
    return fig, ax

def plot_kernel_genper(kernel, genesis_fn, persistence_fn, show_combined=False, \
        my_cmap=plt.get_cmap("magma"), titles=["Kernel", "Growth Function"], \
        my_xrange=[[-0.1, 1.1], [-0.1, 1.1]], vmin=0, vmax=1, invert=True):
    """
    Plot the neighborhood kernel and continuous growth/update function
    """
    
    fig, ax = plt.subplots(1, 3, figsize=(3.0,0.8), gridspec_kw={'width_ratios': [0.1, 1, 2.5]}, facecolor="white")

    my_colorbar = (np.arange(1024, 0, -1) / 1023).reshape(-1, 1) * np.ones((1024, 128))

    my_colorbar = my_cmap(my_colorbar)[:,:,:3]
    if invert:
        my_colorbar = 1.0 - my_colorbar
    subplot_b = ax[0].imshow(my_colorbar)

    ax[0].set_yticks([1023, 511, 0])
    ax[0].set_yticklabels([0., f"{vmax*0.5:.2e}", f"{vmax:.2e}"], {"fontsize": 6})
    ax[0].set_xticklabels("")

    kernel = kernel / kernel.max()


    if invert:
        display_kernel = 1.0 - my_cmap(np.pad(kernel,((1,1),(1,1))))[:,:,0:3]
    else:
        display_kernel = my_cmap(kernel)[:,:,0:3]


    subplot_a = ax[1].imshow(display_kernel, interpolation="nearest") 
    ax[1].set_title(titles[0], fontsize=8)
    ax[1].set_xticklabels('')
    ax[1].set_yticklabels('')
        
    x0 = np.arange(my_xrange[0][0], my_xrange[0][1], (my_xrange[0][1]-my_xrange[0][0]) / 10000)
    x1 = np.arange(my_xrange[1][0], my_xrange[1][1], (my_xrange[1][1]-my_xrange[1][0]) / 10000)
    
    line_color_0 = np.array(my_cmap(128))[:3]
    line_color_1 = np.array(my_cmap(196))[:3]
    if invert:
        line_color_0 = 1.0 - line_color_0
        line_color_1 = 1.0 - line_color_1


    ax_twin = ax[2].twinx()
    if show_combined:
        combined_line_color = np.clip((line_color_0 + line_color_1)/2, 0, 1.)
        ax_twin.plot(x0, genesis_fn(x0) / 2 - 0.5, "--", lw=1, color=line_color_0, alpha=0.2, label="$G_{gen}(\cdot)$")
        ax_twin.plot(x1, persistence_fn(x1) / 2 - 0.5, "-", lw=1, color=line_color_1, alpha=0.2, label="$P(\cdot)$")
        ax_twin.plot(x0, genesis_fn(x0)/2 + persistence_fn(x0)/2, ":", lw=1, color=combined_line_color, alpha=0.5, \
                label="Combined")
    else:

        ax_twin.plot(x0, genesis_fn(x0), "--", lw=1, color=line_color_0, alpha=0.7, label="$G_{gen}(\cdot)$")
        ax_twin.plot(x1, persistence_fn(x1), "-", lw=1, color=line_color_1, alpha=0.7, label="$P(\cdot)$")

    ax_twin.set_yticks([-1.0, 0.0, 1.0])
    ax_twin.set_yticklabels([-1.0, 0.0, 1.0], {"fontsize": 6})

    ax[2].set_yticklabels("")

    ax[2].set_xticks([0.0, 1.0])
    ax[2].set_xticklabels([0.0, 1.0], {"fontsize": 6})
    ax[2].set_title(titles[1], fontsize=8)


    ax[2].set_title(titles[1], fontsize=8)
    ax_twin.legend(fontsize=7)    
    
    plt.tight_layout()
    
    return fig, ax

def get_plot_grid_nbhd_update_next():
    def plot_grid_nbhd_update_next(grid, nbhd, update, next_grid, \
        my_cmap=plt.get_cmap("magma"), titles=None, vmin=0.0, vmax=1):
    
        global subplot_0
        global subplot_1
        global subplot_2
        global subplot_3
        
        if titles == None:
            titles = ["CA grid time t", "Neighborhood", "Update", "CA grid time t+1"]
        
        fig = plt.figure(figsize=(12,12), facecolor="white")
        plt.subplot(221)
        subplot_0 = plt.imshow(grid, cmap=my_cmap, vmin=vmin, vmax=vmax, interpolation="nearest") 
        plt.title(titles[0], fontsize=18)
        
        plt.subplot(222)
        subplot_1 = plt.imshow(nbhd, cmap=my_cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        plt.title(titles[1], fontsize=18)
        
        plt.subplot(223)
        subplot_2 = plt.imshow(update, cmap=my_cmap, interpolation="nearest")
        plt.colorbar()
        plt.title(titles[2], fontsize=18)
        
        plt.subplot(224)
        subplot_3 = plt.imshow(next_grid, cmap=my_cmap, interpolation="nearest")
        plt.title(titles[3], fontsize=18)
        
        plt.tight_layout()
        
        return fig

    return plot_grid_nbhd_update_next

def make_gif(frames_path="./assets/gif_frames/", gif_path="./assets", \
        tag="no tag"):
    
    dir_list = os.listdir(frames_path)

    frames = []

    dir_list.sort()
    for filename in dir_list:
        if "png" in filename:

            image_path = os.path.join(frames_path, filename)
            frames.append(Image.open(image_path))

    assert len(frames) > 1, "no frames to make gif"

    first_frame = frames[0]
    
    gif_id = int((time.time() % 1)*1000)

    gif_path = os.path.join(gif_path, f"gif_{tag}_{gif_id:04d}.gif") 

    first_frame.save(gif_path, format="GIF", append_images=frames, \
            save_all=True, duration=200, loop=0)

    rm_path = os.path.join(frames_path, "*png")

    os.system(f"rm {rm_path}")



def get_smooth_interval(rules_list):
    """
    Convert a single list of Life-like CA rules to their continuous equivalent intervals.
    Note that this only converts one list of rules at a time; birth and survival rules must be
    each converted in turn.
    """

    b = [[(2*bb-1)/18., (2*bb+1)/18.] for bb in rules_list]

    return b

def get_glider():

    ## Life B3/S23
    glider = np.zeros((8,8))
    glider[1, 2] = 1
    glider[2, 1:3] = 1
    glider[3, 1] = 1
    glider[3, 3] = 1

    return glider

def get_puffer():

    ## Move/Morley (B368/S245)
    common_puffer = np.zeros((4, 4))
    common_puffer[0:4, 0] = 1
    common_puffer[1:3, 1] = 1

    return common_puffer

def get_kernel(kernel_fn, kernel_args, radius=13):
    
    radius_x2 = radius * 2
    
    rr = np.sqrt([[((xx - radius)**2 + (yy - radius)**2) 
            for xx in range(radius_x2+1)] for yy in range(radius_x2+1)]) / radius
    
    kernel = gaussian(rr, **kernel_args)
    
    return kernel / np.sum(kernel)


def gaussian(u, **kwargs):
        
    mu = kwargs["mu"]
    
    sigma = kwargs["sigma"]
    
    return np.exp(- ((u - mu) / sigma)**2 / 2 )

def sigmoid_1(x, mu, alpha, gamma=1):
    return 1 / (1 + np.exp(-4 * (x - mu) / alpha))

def get_smooth_steps_fn(**kwargs):
    
    a = kwargs["a"]
    alpha = kwargs["alpha"]
    
    def smooth_steps_fn(x):
        
        result = np.zeros_like(x)
        for edges in a:
            result += sigmoid_1(x, edges[0], alpha) * (1 - sigmoid_1(x, edges[1], alpha))
        
        return result
    
    return smooth_steps_fn

def get_conv_fn(kernel):
    
    def conv_fn(x):
        
        #return  scipy.signal.convolve(x, kernel, mode="same")#, boundary="wrap")
        return scipy.signal.convolve2d(x, kernel, mode='same', boundary='wrap')
    
    return conv_fn
