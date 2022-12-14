import unittest

import numpy as np

import torch

from yuca.ca.neural import NCA


class TestNCA(unittest.TestCase):

    def setUp(self):
        pass

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

        self.assertNotIn(False, params.round(4) == params_again.round(4))

        ca.default_init()

        params = ca.get_params() 
        params = np.random.rand(*params.shape)

        ca.set_params(params)

        params_again = ca.get_params() 

        self.assertNotIn(False, params.round(4) == params_again.round(4))

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


if __name__ == "__main__": #pragma: no cover

    unittest.main()