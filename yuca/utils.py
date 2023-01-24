import os

import time
import numpy as np

import torch

import matplotlib

import matplotlib.pyplot as plt

import skimage 
import skimage.io as sio
import PIL
from PIL import Image

my_cmap = plt.get_cmap("magma")

def seed_all(my_seed=13):

    torch.manual_seed(my_seed)
    np.random.seed(my_seed)

def query_kwargs(key, default, **kwargs):

   return kwargs[key] if key in kwargs.keys() else default

def get_mask(grid, radius):

    mask = np.ones_like(grid)

    mask[grid < radius] *= 0.0 

    return mask

def get_bite_mask(img, bite_radius, coords=None):

    radius = img.shape[2] // 2

    if coords is None:
        coords = np.random.rand(2) * radius

    xx, yy = np.meshgrid(\
            np.arange(-radius, radius + 1, (2 * radius + 1)/ (2*radius)), \
            np.arange(-radius, radius + 1, (2 * radius + 1)/ (2*radius)))

    grid = np.sqrt((xx-coords[0])**2 + (yy-coords[1])**2) / radius

    mask = get_mask(grid, bite_radius) 
    bite_mask = np.zeros((1, img.shape[1], img.shape[2], img.shape[3]))

    for ii in range(img.shape[1]):
        bite_mask[:, ii, :, :] = mask
    
    return bite_mask

def get_aperture(img, aperture_radius):
    
    if aperture_radius < 0.975:
        aperture = 1.0 - get_bite_mask(img, aperture_radius, [0.0, 0.0])
    else: 
        aperture = np.ones_like(img)

    return aperture

def prep_input(img, external_channels=None, \
        batch_size=32, aperture_radius=1.0, bite_radius=0.1, noise_level=0.05):

    external_channels = img.shape[0] if external_channels is None \
            else external_channels

    batch = torch.zeros(batch_size, external_channels, img.shape[2], img.shape[3])

    for ii in range(batch_size):

        my_noise = noise_level \
                * np.random.rand(1, img.shape[1], img.shape[2], img.shape[3])
        bite_mask = get_bite_mask(img, bite_radius)
        aperture = get_aperture(img, aperture_radius)

        batch[ii:ii+1, :img.shape[1], :, :] = torch.tensor(img * aperture * bite_mask \
                + my_noise)


    return batch

def make_target(img):

    target = torch.tensor(img).reshape(1, img.shape[0], \
            img.shape[1], img.shape[2])

    return target

def plot_grid_nbhd(grid, nbhd, update, my_cmap=plt.get_cmap("magma"), \
        titles=None, vmin=0, vmax=1):
    
    global subplot_0
    global subplot_1
    global subplot_2
    global subplot_3
    
    if titles == None:
        titles = ["CA grid", "Neighborhood", "Update"]
    
    fig = plt.figure(figsize=(7,7))
    plt.subplot(221)
    subplot_0 = plt.imshow(grid, cmap=my_cmap, vmin=vmin, vmax=vmax, interpolation="nearest") 
    plt.title(titles[0], fontsize=18)
    
    plt.subplot(222)
    subplot_1 = plt.imshow(nbhd, cmap=my_cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.title(titles[1], fontsize=18)
    
    plt.subplot(223)
    subplot_2 = plt.imshow(update, cmap=my_cmap, interpolation="nearest")
    plt.title(titles[2], fontsize=18)
    
    
    #plt.subplot(224)
    #subplot_3 = plt.imshow(kernel, cmap=my_cmap, interpolation="nearest")
    #plt.title("Kernel", fontsize=18)
    
    plt.tight_layout()
    
    return fig


def plot_kernel_growth(kernel, growth_fn, my_cmap=plt.get_cmap("magma"), \
        titles=["Kernel", "Growth Function"], my_xrange=[-0.1, 1.1], vmin=0, vmax=1):
    
    global subplot_0
    global subplot_1
    
    fig, ax = plt.subplots(1, 2, figsize=(12,4), gridspec_kw={'width_ratios': [1,3]})
    subplot_a = ax[0].imshow(kernel, cmap=my_cmap, vmin=vmin, vmax=vmax, interpolation="nearest") 
    ax[0].set_title(titles[0], fontsize=18)
    
    x = np.arange(my_xrange[0], my_xrange[1], (my_xrange[1]-my_xrange[0]) / 1000)
    
    subplot_b = ax[1].plot(x, growth_fn(x), color=my_cmap(96))
    ax[1].set_title(titles[1], fontsize=18)
    
    plt.tight_layout()
    
    return fig

def make_gif(frames_path="./assets/gif_frames/", gif_path="./assets", \
        tag="no tag"):
    
    dir_list = os.listdir(frames_path)

    frames = []

    dir_list.sort()
    for filename in dir_list:
        if "png" in filename:

            image_path = os.path.join(frames_path, filename)
            frames.append(Image.open(image_path))

    try: 
        assert len(frames) > 1, "no frames to make gif"
    except:
        import pdb; pdb.set_trace()

    first_frame = frames[0]
    
    gif_id = int((time.time() % 1)*1000)

    if os.path.exists(gif_path):
        pass
    else:
        os.mkdir(gif_path)

    gif_path = os.path.join(gif_path, f"gif_{tag}_{gif_id:04d}.gif") 

    first_frame.save(gif_path, format="GIF", append_images=frames, \
            save_all=True, duration=42, loop=0)

    rm_path = os.path.join(frames_path, "*png")

    os.system(f"rm {rm_path}")

def save_fig_sequence(grid, ca, num_steps=10, \
        frames_path="./assets/gif_frames", \
        gif_path="./assets", mode=0, tag="no_tag", \
        speedup=1,\
        cmap=plt.get_cmap("magma"), invert_colors=True):

    """
    mode == 0 -- save state grid

    modes 1 through 4 not implemented yet
    mode == 1 -- save state grid and neighborhoods
    mode == 2 -- save state grid and update grid
    mode == 3 -- save state grid, neighborhood grid, and update grid
    mode == 4 -- save grid, state grid, neighborhood grid, and update grid
    """

    old_device = ca.my_device 
    #ca.to_device("cpu")
    grid = grid.to(ca.my_device)

    # number of leading zeros
    leading = num_steps // 10 + 1

    if not os.path.exists(frames_path):
        my_cmd = f"mkdir -p {frames_path}"
        os.system(my_cmd)

    if type(tag) is str:
        gif_path = os.path.join(gif_path, tag)
    else:
        gif_path = os.path.join(gif_path, tag[0])
        tag = tag[1]

    for step in range(num_steps):
        
        if grid.max() < 0.001:
            break

        if leading <= 5:
            save_path = os.path.join(frames_path, f"frame_{step:05d}.png")
        else: 
            save_path = os.path.join(frames_path, f"frame_{step:010}.png")

        if grid.shape[1] >= 4:
            img = np.uint8(255 * grid[0,0:4,:,:].detach().cpu().numpy())
            img = img.transpose(1,2,0)
        elif grid.shape[1] == 3:
            img = np.uint8(255 * grid[0,0:3,:,:].detach().cpu().numpy())
            img = img.transpose(1,2,0)
        elif grid.shape[1] == 2:

            img_grid = np.zeros((3,grid.shape[-2], grid.shape[-1]),\
                    dtype=np.uint8)
            img_grid[1,:,:] = np.uint8(255 * grid[0,0,:,:].detach().cpu().numpy())
            img_grid[2,:,:] = np.uint8(255 * grid[0,1,:,:].detach().cpu().numpy())

            img = img_grid.transpose(1,2,0)
        else:
            img = np.uint8(255 * cmap(grid[0,0,:,:].detach().cpu().numpy()))

        if invert_colors:
            img[:,:,:3] = 255 - img[:,:,:3]

        sio.imsave(save_path, img, check_contrast=False)

        for speed_step in range(speedup):
            grid = ca.get_frame(grid, mode = mode)


    if step > 0:
        make_gif(frames_path=frames_path, gif_path=gif_path, tag=tag) 
    else: 
        print("vanishing grid, no gif made")

    os.system(f"rm -r {frames_path}")
    #ca.to_device(old_device)
