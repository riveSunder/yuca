import os
import time

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from yuca.multiverse import CA
from yuca.zoo.librarian import Librarian
from yuca.cppn import CPPN

import bokeh
from bokeh.io import curdoc
import bokeh.io as bio
from bokeh.io import output_notebook, show
from bokeh.plotting import figure

from bokeh.layouts import column, row
from bokeh.models import TextInput, Button, Paragraph
from bokeh.models import ColumnDataSource
from bokeh.events import DoubleTap, Tap



global grid
global my_period
global my_step

my_step = 0
my_period = 512
grid = torch.zeros(1,1,96, 96)


# instantiate librarian (zoo manager) and ca (simulator)
lib = Librarian()
ca = CA() 
ca.no_grad()

pattern_index = lib.index
pattern, rule_string = lib.load("frog001")

place_h = (grid.shape[-2] - pattern.shape[-2]) // 2 - 3
place_w = (grid.shape[-1] - pattern.shape[-1]) // 2 - 3

grid[:,:, place_h:place_h+pattern.shape[-2], place_w:place_w+pattern.shape[-1]] = \
        torch.tensor(pattern[:,:,:,:])

ca.restore_config(rule_string)

p = figure(plot_width=768, plot_height=768, title="CA Universe")

#p_plot = figure(plot_width=int(1.25*256), plot_height=int(1.25*256), title="'Reward'")
    
source = ColumnDataSource(data=dict(my_image=[grid.squeeze().cpu().numpy()]))
#source_plot = ColumnDataSource(data=dict(x=np.arange(1), y=np.arange(1)*0))

img = p.image(image='my_image',x=0, y=0, dw=256, dh=256, palette="Magma256", source=source)
#line_plot = p_plot.line(line_width=3, color="firebrick", source=source_plot)

button_go = Button(sizing_mode="stretch_width", label="Run >")     
button_slower = Button(sizing_mode="stretch_width",label="<< Slower")
button_faster = Button(sizing_mode="stretch_width",label="Faster >>")

button_reset_prev_pattern = Button(sizing_mode="stretch_width",label="Previous pattern")

button_reset_this_pattern = Button(sizing_mode="stretch_width",label="Reset current pattern")

button_reset_next_pattern = Button(sizing_mode="stretch_width",label="Next pattern")

message = Paragraph()

def update():
        global grid
        global my_step
        global ca
        #global rewards
        global pattern_index
        
        ca.no_grad()
        grid = ca(grid)
        
        my_img = grid.squeeze().cpu().numpy()
        new_data = dict(my_image=[my_img])
        
        #new_line = dict(x=np.arange(my_step+2), y=rewards)
        #new_line = dict(x=[my_step], y=[r.cpu().numpy().item()])

        source.stream(new_data, rollover=0)
        #source_plot.stream(new_line, rollover=2000)

        my_step += 1
        message.text = f"Message! step {my_step}, period: {my_period} ms"
    
def go():
   
    if button_go.label == "Run >":
        my_callback = curdoc().add_periodic_callback(update, my_period)
        button_go.label = "Pause"
        
    else:
        curdoc().remove_periodic_callback(curdoc().session_callbacks[0])
        button_go.label = "Run >"

def faster():
    global my_period
    my_period = max([my_period / 2, 128])
    go()
    time.sleep(my_period*0.001)
    go()
    
def slower():
    global my_period
    my_period = min([my_period * 2, 8192])
    go()
    time.sleep(my_period*0.001)
    go()

def reset_this_pattern():

    global grid
    global lib
    global ca

    my_step = 0
    
    grid = torch.zeros(1, 1, 96, 96) 

    temp_pattern_name = lib.index.pop(-1)

    pattern, rule_string = lib.load(temp_pattern_name)
    lib.index.append(temp_pattern_name)

    ca.restore_config(rule_string)
    place_h = (grid.shape[-2] - pattern.shape[-2]) // 2 - 3
    place_w = (grid.shape[-1] - pattern.shape[-1]) // 2 - 3

    grid[:,:, place_h:place_h+pattern.shape[-2], \
            place_w:place_w+pattern.shape[-1]] = torch.tensor(pattern[:,:,:,:])

    new_data = dict(my_image=[(grid.squeeze()).cpu().numpy()])
    
    source.stream(new_data, rollover=0)
    #source_plot.stream(new_line, rollover=2000)

 
def reset_next_pattern():
    
    global grid
    global lib
    global ca

    my_step = 0
    
    grid = torch.zeros(1, 1, 96, 96) 

    temp_pattern_name = lib.index.pop(0)

    pattern, rule_string = lib.load(temp_pattern_name)

    ca.restore_config(rule_string)
    place_h = (grid.shape[-2] - pattern.shape[-2]) // 2 - 3
    place_w = (grid.shape[-1] - pattern.shape[-1]) // 2 - 3

    grid[:,:, place_h:place_h+pattern.shape[-2], \
            place_w:place_w+pattern.shape[-1]] = torch.tensor(pattern[:,:,:,:])

    lib.index.append(temp_pattern_name)
            
    new_data = dict(my_image=[(grid.squeeze()).cpu().numpy()])
    
    source.stream(new_data, rollover=o)
    #source_plot.stream(new_line, rollover=2000)
    message.text = f"reset next"
        
def reset_prev_pattern():
   
    global grid
    global lib
    global ca

    my_step = 0
    
    grid = torch.zeros(1, 1, 96, 96) 

    temp_pattern_name = lib.index.pop(-1)
    lib.index.insert(0, temp_pattern_name)
    temp_pattern_name = lib.index.pop(-1)

    pattern, rule_string = lib.load(temp_pattern_name)

    ca.restore_config(rule_string)
    place_h = (grid.shape[-2] - pattern.shape[-2]) // 2 - 3
    place_w = (grid.shape[-1] - pattern.shape[-1]) // 2 - 3

    grid[:,:, place_h:place_h+pattern.shape[-2], \
            place_w:place_w+pattern.shape[-1]] = torch.tensor(pattern[:,:,:,:])

    lib.index.append(temp_pattern_name)
            
    new_data = dict(my_image=[(grid.squeeze()).cpu().numpy()])
    
    source.stream(new_data, rollover=o)
    #source_plot.stream(new_line, rollover=2000)
    message.text = f"reset prev"

#def human_toggle(event):
#    global action
#
#    coords = [np.round(env.height*event.y/256-0.5), np.round(env.width*event.x/256-0.5)]
#    offset_x = (env.height - env.action_height) / 2
#    offset_y = (env.width - env.action_width) / 2
#
#    coords[0] = coords[0] - offset_x
#    coords[1] = coords[1] - offset_y
#
#    coords[0] = np.uint8(np.clip(coords[0], 0, env.action_height-1))
#    coords[1] = np.uint8(np.clip(coords[1], 0, env.action_height-1))
#
#    action[:, :, coords[0], coords[1]] = 1.0 * (not(action[:, :, coords[0], coords[1]]))
#
#    padded_action = stretch_pixel/2 + env.inner_env.action_padding(action).squeeze()
#
#    my_img = (padded_action*2 + obs.squeeze()).cpu().numpy()
#    my_img[my_img > 3.0] = 3.0
#    (padded_action*2 + obs.squeeze()).cpu().numpy()
#    new_data = dict(my_image=[my_img])
#
#    source.stream(new_data, rollover=8)
##

reset_this_pattern()
     
#p.on_event(Tap, human_toggle)
#p.on_event(DoubleTap, clear_toggles)

button_reset_prev_pattern.on_click(reset_prev_pattern)
button_reset_this_pattern.on_click(reset_this_pattern)
button_reset_next_pattern.on_click(reset_next_pattern)

button_go.on_click(go)
button_faster.on_click(faster)
button_slower.on_click(slower)

display_layout = row(p) #, column(p_plot, p_weights))
control_layout = row(button_slower, button_go, button_faster)
reset_layout = row(button_reset_prev_pattern, \
        button_reset_this_pattern, \
        button_reset_next_pattern)
        
message_layout = row(message)

curdoc().add_root(display_layout)
curdoc().add_root(control_layout)
curdoc().add_root(reset_layout)

curdoc().add_root(message_layout)
