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



if __name__ == "__main__":

    unittest.main()
