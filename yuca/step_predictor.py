import os
import argparse

import time

import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F


import yuca
from yuca.multiverse import CA
from yuca.utils import query_kwargs, get_bite_mask, save_fig_sequence, seed_all


class StepPredictor(nn.Module):

    def __init__(self, **kwargs):
        super(StepPredictor, self).__init__()

        self.layer_repeats = query_kwargs("layer_repeats", 1, **kwargs)
        self.kernel_radius = query_kwargs("kernel_radius", 13, **kwargs)
        self.hidden_channels = query_kwargs("hidden_channels", 8, **kwargs)
        self.out_channels = query_kwargs("out_channels", 1, **kwargs)
        self.dropout = query_kwargs("dropout", 0.1, **kwargs)
        self.my_device = query_kwargs("device", "cpu", **kwargs)

        self.my_device = torch.device(self.my_device)
        
        seed = query_kwargs("seed", 42, **kwargs)
        tag = query_kwargs("tag", "none", **kwargs)

        seed_all(seed)

        self.init_model()

        self.to(self.my_device)

        exp_fingerprint = str(int(time.time() * 1000))[-6:]

        self.exp_tag = f"{tag}_l_{self.layer_repeats}_kr_{self.kernel_radius}"\
                f"_hid{self.hidden_channels}_seed{seed}_{exp_fingerprint}"
        
    def init_model(self):

        my_padding = self.kernel_radius

        self.conv_layers = []
        

        for ii in range(self.layer_repeats):

            self.conv_layers.append(nn.Conv2d(in_channels=self.out_channels, \
                    out_channels=self.hidden_channels,\
                    kernel_size=2 * self.kernel_radius+1, stride=1, \
                    padding=self.kernel_radius))

            for gg, param in enumerate(self.conv_layers[-1].parameters()):
                self.register_parameter(f"layer{ii}_param{gg}", param)

            self.conv_layers.append(nn.Conv2d(in_channels=self.hidden_channels,\
                    out_channels=self.out_channels, kernel_size=1))

            for hh, param in enumerate(self.conv_layers[-1].parameters()):
                self.register_parameter(f"layer{ii}_1x1_param{hh}", param)
                    
    def forward(self, x):

        for jj in range(0, self.layer_repeats, 2):
            
            x0 = torch.relu(self.conv_layers[jj](x))
            x1 = F.dropout(x0, p = self.dropout, training=self.training)

            x = x + torch.sigmoid(self.conv_layers[jj + 1](x1))

        return x

            
    def fit(self, train_dataloader, val_dataloader=None, **kwargs):

        max_epochs = query_kwargs("max_epochs", 100, **kwargs)
        disp_every = query_kwargs("disp_every", max([1, max_epochs // 10]), **kwargs)
        lr = query_kwargs("lr", 3e-4, **kwargs)

        self.train()

        smooth_loss = []
        epochs = []
        val_losses = []
        wall_time_v = []
        wall_time_t = []

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        t0 = time.time()
        for epoch in range(max_epochs):

            if epoch % disp_every == 0:
                self.eval()
                val_loss = 0.0
                val_samples = 0

                with torch.no_grad():
                    for batch_x, batch_y in val_dataloader:

                        pred = model(batch_x.to(self.my_device))

                        #mse loss
                        val_loss += torch.sum((batch_y.to(self.my_device) - pred)**2)

                        val_samples += batch_y.shape[0]

                    val_loss /= val_samples

                val_losses.append(val_loss.item())
                epochs.append(epoch)
                wall_time_v.append(time.time() - t0)

                if len(smooth_loss):
                    my_smooth_loss = smooth_loss[-1]
                else:
                    my_smooth_loss = "none"

                print(f"loss at epoch {epoch} val: {val_loss}, "\
                        f"smooth train: {my_smooth_loss} "\
                        f"time elapsed: {wall_time_v[-1]:.3f}")

                progress = {"wall_time_t": wall_time_t,\
                        "train_smooth_loss": smooth_loss,\
                        "wall_time_v": wall_time_v,\
                        "epochs": epochs,\
                        "val_loss": val_losses}

                save_path = os.path.split(os.path.split(\
                        os.path.realpath(__file__))[0])[0]

                save_path = os.path.join(save_path, "logs", f"{self.exp_tag}.npy")

                np.save(save_path, progress)

            self.train()
            for batch_x, batch_y in train_dataloader:

                optimizer.zero_grad()

                pred = model(batch_x.to(self.my_device))

                loss = torch.mean((batch_y.to(self.my_device) - pred)**2)

                loss.backward()
                optimizer.step()
                
                if len(smooth_loss):
                    smooth_loss.append(smooth_loss[-1] * 0.99 \
                            + loss.detach().item() * 0.01)

                else:
                    smooth_loss.append(loss.detach().item())

                wall_time_t.append(time.time() - t0)

    def evaluate(self, test_dataloader):

        self.eval()
        with torch.no_grad():

            test_loss = 0.0
            test_samples = 0

            for batch_x, batch_y in test_dataloader:

                pred = model(batch_x.to(self.my_device))

                #mse loss
                test_loss += torch.sum((batch_y.to(self.my_device) - pred)**2)

                test_samples += batch_y.shape[0]

            test_loss /= test_samples


        return test_loss.item()

    def end_evaluate(self, train_dataloader, val_dataloader, test_dataloader):

        train_loss = self.evaluate(train_dataloader)
        val_loss = self.evaluate(val_dataloader)
        test_loss = self.evaluate(test_dataloader)

        msg =   f"{self.exp_tag} train loss: {test_loss} \n" \
                f"{self.exp_tag} val loss: {test_loss} \n" \
                f"{self.exp_tag} test loss: {test_loss} \n"

        print(msg)

        save_path = os.path.split(os.path.realpath(__file__))[0]
        save_path = os.path.join(save_path, "logs", f"{self.exp_tag}_final.txt")


        with open(save_path) as f:
            f.write(msg)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_filepath", type=str, \
            default="data/geminium_test_dataset_x.pt", help="filepath to input data" )

    parser.add_argument("-t", "--targets_filepath", type=str, \
            default="data/geminium_test_dataset_y.pt", \
            help="filepath to target data" )

    parser.add_argument("-d", "--device", type=str, \
            default="cpu", help="device to use (cpu or cuda)")

    parser.add_argument("-m", "--max_epochs", type=int, default=10)

    parser.add_argument("-s", "--seed", type=int, nargs="+", default=13)

    args = parser.parse_args()
    #my_kwargs = dict(args._get_kwargs())

    max_epochs = args.max_epochs
    kernel_radius = 31
    hidden_channels = 8
    batch_size = 512
    pin_memory = True if "cuda" in args.device else False

    if type(args.seed) is not list:
        args.seed = [args.seed]

    for seed in args.seed:
        for layer_repeats in [1, 2, 4]:

            data = torch.load(args.input_filepath)
            target = torch.load(args.targets_filepath)

            train_split = data.shape[0] // 10

            val_data = torch.load(args.input_filepath)[0:train_split]
            val_target = torch.load(args.targets_filepath)[0:train_split]

            train_data = torch.load(args.input_filepath)[train_split:]
            train_target = torch.load(args.targets_filepath)[train_split:]

            train_dataset = torch.utils.data.TensorDataset(train_data, \
                    train_target)

            val_dataset = torch.utils.data.TensorDataset(val_data, \
                    val_target)

            train_dataloader = torch.utils.data.DataLoader(train_dataset, \
                    batch_size=batch_size, \
                    shuffle=True, \
                    pin_memory=pin_memory, \
                    num_workers=24)

            val_dataloader = torch.utils.data.DataLoader(val_dataset, \
                    batch_size=batch_size, \
                    pin_memory=pin_memory, \
                    num_workers=24)


            tag = os.path.splitext(os.path.split(args.input_filepath)[-1])[0][0:8]
            model = StepPredictor(layer_repeats=layer_repeats, \
                    kernel_radius=kernel_radius, \
                    hidden_channels=hidden_channels, \
                    tag=tag,
                    seed=seed, device=args.device)

            model.fit(train_dataloader, val_dataloader=val_dataloader, 
                    max_epochs=max_epochs)

            test_data = torch.load(args.input_filepath.replace("train", "test"))
            test_target = torch.load(args.targets_filepath.replace("train", "test"))

            test_dataset = torch.utils.data.TensorDataset(test_data, \
                    test_target)

            test_dataloader = torch.utils.data.DataLoader(test_dataset, \
                    batch_size=batch_size, \
                    pin_memory=pin_memory, \
                    num_workers=24)

            model.end_evaluate(train_dataloader, val_dataloader, test_dataloader)
