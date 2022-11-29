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

    def test_multiverse_id_layer(self):

        ca = NCA()
        ca.default_init()

        input_x = torch.randn(1, 1, 32, 32)

        id_x = ca.id_layer(input_x)

        self.assertNotIn(False, input_x == id_x)

    def test_multiverse_id_conv(self):

        ca = NCA()
        ca.default_init()

        input_x = torch.randn(1, 1, 32, 32)

        id_x = ca.id_conv(input_x)

        self.assertNotIn(False, input_x == id_x)

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
