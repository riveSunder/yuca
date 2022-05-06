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
pattern, pattern_meta = lib.load("orbium_orbium000")
rule_string = pattern_meta["ca_config"]
entry_point = pattern_meta["entry_point"]
commit_hash = pattern_meta["commit_hash"]
notes = pattern_meta["notes"]

place_h = (grid.shape[-2] - pattern.shape[-2]) // 2 - 3
place_w = (grid.shape[-1] - pattern.shape[-1]) // 2 - 3

grid[:,:, place_h:place_h+pattern.shape[-2], place_w:place_w+pattern.shape[-1]] = \
        torch.tensor(pattern[:,:,:,:])

ca.restore_config(rule_string)

p = figure(plot_width=384, plot_height=384, title="CA Universe")

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

message = Paragraph(sizing_mode="scale_width")
pattern_message = Paragraph(sizing_mode="scale_width")
config_message = Paragraph(sizing_mode="scale_width")
entry_message = Paragraph(sizing_mode="scale_width")
git_message = Paragraph(sizing_mode="scale_width")

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
        message.text = f"Step {my_step}, period: {my_period} ms"
    
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

    pattern, pattern_meta =lib.load(temp_pattern_name)
    rule_string = pattern_meta["ca_config"]
    entry_point = pattern_meta["entry_point"]
    commit_hash = pattern_meta["commit_hash"]
    notes = pattern_meta["notes"]

    lib.index.append(temp_pattern_name)

    ca.restore_config(rule_string)
    place_h = (grid.shape[-2] - pattern.shape[-2]) // 2 - 3
    place_w = (grid.shape[-1] - pattern.shape[-1]) // 2 - 3

    grid[:,:, place_h:place_h+pattern.shape[-2], \
            place_w:place_w+pattern.shape[-1]] = torch.tensor(pattern[:,:,:,:])

    new_data = dict(my_image=[(grid.squeeze()).cpu().numpy()])
    
    source.stream(new_data, rollover=0)
    #source_plot.stream(new_line, rollover=2000)

    ca_name = os.path.splitext(rule_string)[0]

    pattern_message.text = f"(unofficial) moniker: {temp_pattern_name}"
    config_message.text = f"\nCA config: {ca_name} " 
    entry_message.text =  f"\nevolved with: \n{entry_point}" 

    my_git_message = f"\nat git commit hash: \n{commit_hash}"
    if len(notes):
        my_git_message += f"\n\nnote: {notes}"

    git_message.text = my_git_message
 
def reset_next_pattern():
    
    global grid
    global lib
    global ca

    my_step = 0
    
    grid = torch.zeros(1, 1, 96, 96) 

    temp_pattern_name = lib.index.pop(0)

    pattern, pattern_meta =lib.load(temp_pattern_name)
    rule_string = pattern_meta["ca_config"]
    entry_point = pattern_meta["entry_point"]
    commit_hash = pattern_meta["commit_hash"]
    notes = pattern_meta["notes"]

    ca.restore_config(rule_string)
    place_h = (grid.shape[-2] - pattern.shape[-2]) // 2 - 3
    place_w = (grid.shape[-1] - pattern.shape[-1]) // 2 - 3

    grid[:,:, place_h:place_h+pattern.shape[-2], \
            place_w:place_w+pattern.shape[-1]] = torch.tensor(pattern[:,:,:,:])

    lib.index.append(temp_pattern_name)

    new_data = dict(my_image=[(grid.squeeze()).cpu().numpy()])
    
    source.stream(new_data, rollover=0)
    #source_plot.stream(new_line, rollover=2000)
    ca_name = os.path.splitext(rule_string)[0]

    pattern_message.text = f"(unofficial) moniker: {temp_pattern_name}"
    config_message.text = f"\nCA config: {ca_name} " 
    entry_message.text =  f"\nevolved with: {entry_point}" 
    git_message.text = f"\nat git commit hash: \n {commit_hash}"

    if commit_hash == "none":
        reset_next_pattern()
            

        
def reset_prev_pattern():
   
    global grid
    global lib
    global ca

    my_step = 0
    
    grid = torch.zeros(1, 1, 96, 96) 

    temp_pattern_name = lib.index.pop(-1)
    lib.index.insert(0, temp_pattern_name)
    temp_pattern_name = lib.index.pop(-1)

    pattern, pattern_meta =lib.load(temp_pattern_name)
    rule_string = pattern_meta["ca_config"]
    entry_point = pattern_meta["entry_point"]
    commit_hash = pattern_meta["commit_hash"]
    notes = pattern_meta["notes"]

    ca.restore_config(rule_string)
    place_h = (grid.shape[-2] - pattern.shape[-2]) // 2 - 3
    place_w = (grid.shape[-1] - pattern.shape[-1]) // 2 - 3

    grid[:,:, place_h:place_h+pattern.shape[-2], \
            place_w:place_w+pattern.shape[-1]] = torch.tensor(pattern[:,:,:,:])

    lib.index.append(temp_pattern_name)
            
    new_data = dict(my_image=[(grid.squeeze()).cpu().numpy()])
    
    source.stream(new_data, rollover=0)
    #source_plot.stream(new_line, rollover=2000)
    ca_name = os.path.splitext(rule_string)[0]

    pattern_message.text = f"(unofficial) moniker: {temp_pattern_name}"
    config_message.text = f"\nCA config: {ca_name} " 
    entry_message.text =  f"\nevolved with: {entry_point}" 
    git_message.text = f"\nat git commit hash: \n {commit_hash}"

    if commit_hash == "none":
        reset_prev_pattern()


reset_this_pattern()
     
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
pattern_message_layout = row(config_message)
config_message_layout = row(pattern_message)
entry_message_layout = row(entry_message)
git_message_layout = row(git_message)

curdoc().add_root(display_layout)
curdoc().add_root(control_layout)
curdoc().add_root(reset_layout)

curdoc().add_root(message_layout)
curdoc().add_root(pattern_message_layout)
curdoc().add_root(config_message_layout)
curdoc().add_root(entry_message_layout)
curdoc().add_root(git_message_layout)
