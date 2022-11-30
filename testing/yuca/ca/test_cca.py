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

    def test_multiverse_id_layer(self):

        ca = CCA()
        ca.default_init()

        input_x = torch.randn(1, 1, 32, 32)

        id_x = ca.id_layer(input_x)

        self.assertNotIn(False, input_x == id_x)

    def test_multiverse_id_conv(self):

        ca = CCA()
        ca.default_init()

        input_x = torch.randn(1, 1, 32, 32)

        id_x = ca.id_conv(input_x)

        self.assertNotIn(False, input_x == id_x)

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
