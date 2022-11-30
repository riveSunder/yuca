import unittest

import numpy as np
import torch

from yuca.ca.reaction_diffusion import RxnDfn

class TestRxnDfn(unittest.TestCase):

    def setUp(self):
        pass

    def test_forward(self):

        rxn = RxnDfn()

        x = torch.rand(1,2,64,64)

        next_x = rxn(x)

        self.assertEqual(x.shape, next_x.shape)

    def test_multiverse_set_params(self):

        rxn = RxnDfn()
        rxn.default_init()

        params = rxn.get_params() 
        params = np.random.rand(*params.shape)

        rxn.set_params(params)

        params_again = rxn.get_params() 

        self.assertNotIn(False, params.round(4) == params_again.round(4))

        rxn.default_init()

        params = rxn.get_params() 
        params = np.random.rand(*params.shape)

        rxn.set_params(params)

        params_again = rxn.get_params() 

        self.assertNotIn(False, params.round(4) == params_again.round(4))


if __name__ == "__main__":

    unittest.main()
