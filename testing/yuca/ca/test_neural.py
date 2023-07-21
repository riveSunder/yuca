import os
import unittest


import numpy as np

import torch

from yuca.ca.neural import NCA


class TestNCA(unittest.TestCase):

    def setUp(self):
        pass

    def test_configs(self):

        nca_1 = NCA()
        nca_1.no_grad()

        my_filepath = "./temp_delete_nca_config.npy"

        my_config_1 = nca_1.make_config()
        my_config_1_2 = nca_1.make_config()

        nca_1.save_config(my_filepath)
        nca_1.restore_config(my_filepath)
        my_config_2 = nca_1.make_config()

        nca_2 = NCA()
        nca_2.no_grad()
        my_config_3 = nca_2.make_config()

        nca_2.restore_config(my_filepath)
        my_config_4 = nca_2.make_config()

        # 1 == 3; 1 != 2

        for key in my_config_1.keys():
            self.assertIn(key, my_config_2.keys())
            self.assertIn(key, my_config_3.keys())

            if key == "params":
            
                sae_1_1b = np.sum(np.abs(my_config_1[key] - my_config_1_2[key]))

                sae_1_2 = np.sum(np.abs(my_config_1[key] - my_config_2[key]))
                sae_2_3 = np.sum(np.abs(my_config_2[key] - my_config_3[key]))
                sae_1_3 = np.sum(np.abs(my_config_1[key] - my_config_3[key]))
                sae_1_4 = np.sum(np.abs(my_config_1[key] - my_config_4[key]))

                self.assertEqual(0.0, sae_1_1b)
                self.assertEqual(0.0, sae_1_2)
                self.assertNotEqual(0.0, sae_1_3)
                self.assertNotEqual(0.0, sae_2_3)
                self.assertEqual(0.0, sae_1_4)

        os.system(f"rm {my_filepath}")
        
    def test_multiverse_forward(self):

        ca = NCA()
        ca.no_grad()

        ca.default_init()

        for vectorization in [1,2,4,9]:

            ca.to_device("cpu")

            x = torch.randn(vectorization, 1, 32, 32)
            new_x = ca.forward(x)

            self.assertEqual(new_x.shape, x.shape)
            
            if torch.cuda.is_available():

                ca.to_device("cuda")

                x = torch.randn(vectorization, 1, 32, 32).to("cuda")
                new_x = ca.forward(x)

                self.assertEqual(new_x.shape, x.shape)
                
        ca = NCA()
        ca.default_init()

        for vectorization in [1,2,4,9]:

            ca.to_device("cpu")

            x = torch.randn(vectorization, 1, 32, 32)
            new_x = ca.forward(x)

            self.assertEqual(new_x.shape, x.shape)
            
            if (0): #torch.cuda.is_available():

                ca.to_device("cuda")

                x = torch.randn(vectorization, 1, 32, 32).to("cuda")
                new_x = ca.forward(x)

                self.assertEqual(new_x.shape, x.shape)

    def test_id_conv(self):

        for channels in [1,2,4,8,15,16]:
            nca = NCA(internal_channels=channels, \
                    hidden_channels=channels,\
                    external_channels=channels)

            grid = torch.rand(1,channels,32,32)

            id_grid = nca.id_conv(grid)

            sum_difference = (grid - id_grid).sum()

            self.assertAlmostEqual(0.0, sum_difference)

            id_x = nca.id_layer(grid)

            self.assertNotIn(False, grid == id_x)

    def test_multiverse_set_params(self):

        ca = NCA()
        ca.default_init()

        params = ca.get_params() 
        params = np.random.rand(*params.shape)

        ca.set_params(params)

        params_again = ca.get_params() 

        self.assertNotIn(False, params.round(3) == params_again.round(3))

        ca.default_init()

        params = ca.get_params() 
        params = np.random.rand(*params.shape)

        ca.set_params(params)

        params_again = ca.get_params() 

        self.assertNotIn(False, params.round(3) == params_again.round(3))

    def test_multiverse_to(self):

        ca = NCA()
        ca.default_init()
        ca.include_parameters()

        for my_device in ["cpu", torch.device("cpu"), "cuda", torch.device("cuda")]:
            
            if "cuda" in torch.device(my_device).type and torch.cuda.is_available():
                ca.to_device(my_device)
                 
                for ii, genesis_fn in enumerate(ca.genesis_fns):
                    for jj, param in enumerate(genesis_fn.named_parameters()):
                        self.assertEqual(param[1].device.type, \
                                torch.device(my_device).type)

                for kk, persistence_fn in enumerate(ca.persistence_fns):
                    for ll, param in enumerate(persistence_fn.named_parameters()):
                        self.assertEqual(param[1].device.type, \
                                torch.device(my_device).type)

                for mm, param in enumerate(ca.named_parameters()):

                    self.assertEqual(param[1].device.type, \
                            torch.device(my_device).type)
                

        ca = NCA()
        ca.default_init()

        for my_device in ["cpu", torch.device("cpu"), "cuda", torch.device("cuda")]:
            
            if "cuda" in torch.device(my_device).type and torch.cuda.is_available():
                ca.to_device(my_device)
                 
                for ii, genesis_fn in enumerate(ca.genesis_fns):
                    for jj, param in enumerate(genesis_fn.named_parameters()):
                        self.assertEqual(param[1].device.type, \
                                torch.device(my_device).type)

                for kk, persistence_fn in enumerate(ca.persistence_fns):
                    for ll, param in enumerate(persistence_fn.named_parameters()):
                        self.assertEqual(param[1].device.type, \
                                torch.device(my_device).type)

                for mm, param in enumerate(ca.named_parameters()):
                    self.assertEqual(param[1].device.type, \
                            torch.device(my_device).type)

    def test_set_kernel_radius(self):

        nca = NCA()
        nca.no_grad()
        nca.make_config()

        for kr in [13, 17, 29]:

            nca.set_kernel_radius(kr)

            my_config = nca.make_config()

            config_kr = my_config["neighborhood_kernel_config"]["radius"] 
            self.assertEqual(kr, nca.kernel_radius)
            self.assertEqual(kr, config_kr)
            self.assertEqual(nca.kernel_radius * 2 + 1, nca.neighborhood_kernels.shape[-1])

    def test_kernel_params_config(self):

        nca = NCA()

        kparams = 1.0 * nca.kernel_params

        ca_config = nca.make_config()

        kconfig = ca_config["neighborhood_kernel_config"]
        kernel_kwargs = kconfig["kernel_kwargs"]
        kconfig_params = None

        for key in kernel_kwargs.keys():
            if kconfig_params is None:
                kconfig_params = np.array(kernel_kwargs[key])
            else:
                kconfig_params = np.append(kconfig_params, \
                        np.array(kernel_kwargs[key]))

        # should contain the same info
        self.assertEqual(kconfig_params.shape, kparams.shape)

        self.assertEqual(0.0, np.sum(np.abs(np.array(kconfig_params - kparams))))


if __name__ == "__main__": #pragma: no cover

    unittest.main()
