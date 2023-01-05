import os
import unittest

import numpy as np

import torch

from yuca.ca.continuous import CCA


class TestCCA(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_frame(self):

        ca = CCA()
        ca.default_init()

        x = torch.randn(1, 1, 32, 32)
        for frame_mode in [0,1,2,3,4]:
            frame_return = ca.get_frame(x,mode=frame_mode)

            if frame_mode == 0:
                self.assertEqual(1, len(frame_return))
            elif True in (frame_mode == np.array([1,2])):
                self.assertEqual(2, len(frame_return))
            elif True in (frame_mode == np.array([3])):
                self.assertEqual(3, len(frame_return))
            elif True in (frame_mode == np.array([4])):
                self.assertEqual(4, len(frame_return))
            
    def test_configs(self):

        cca_1 = CCA()
        cca_1.no_grad()

        my_filepath = "./temp_delete_cca_config.npy"

        my_config_1 = cca_1.make_config()
        my_config_1_2 = cca_1.make_config()

        cca_1.save_config(my_filepath)
        cca_1.restore_config(my_filepath)
        my_config_2 = cca_1.make_config()

        cca_2 = CCA()
        cca_2.no_grad()
        my_config_3 = cca_2.make_config()

        cca_2.restore_config(my_filepath)
        my_config_4 = cca_2.make_config()

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

    def test_id_conv(self):


        for channels in [1,2,4,8,15,16]:
            cca = CCA(internal_channels=channels, \
                    external_channels=channels)
            grid = torch.rand(1,channels,32,32)

            id_grid = cca.id_conv(grid)

            sum_difference = (grid - id_grid).sum()

            self.assertAlmostEqual(0.0, sum_difference)

            id_x = cca.id_layer(grid)

            self.assertNotIn(False, grid == id_x)


    def test_alive_mask(self):

        ca = CCA(external_channels=4)
        ca.default_init()

        x = torch.randn(1, 4, 32, 32)
        masked = ca(x)

        
    def test_multiverse_forward(self):

        for mode in ["functional", "neurofunctional"]:
            for internal_channels in [1, 8, 16, 4]:

                ca = CCA(ca_mode = mode, internal_channels=internal_channels)
            #ca = CCA(ca_mode = mode)
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
                    
            ca = CCA(ca_mode = mode)
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


    def test_multiverse_set_params(self):

        for ca_mode in ["neural", "functional", "neurofunctional"]:
            ca = CCA(ca_mode=ca_mode)
            ca.default_init()

            params = ca.get_params() 
            params = np.random.rand(*params.shape)

            ca.set_params(params)

            params_again = ca.get_params() 

            self.assertNotIn(False, params.round(4) == params_again.round(4))

            ca.default_init()

            params = ca.get_params() 
            params = np.random.rand(*params.shape)

            ca.set_params(params)

            params_again = ca.get_params() 

            self.assertNotIn(False, params.round(4) == params_again.round(4))

    def test_multiverse_to(self):

        ca = CCA()
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
                

        ca = CCA()
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


if __name__ == "__main__": #pragma: no cover

    unittest.main()
