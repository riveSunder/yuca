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

    def test_update_universe(self):

        rxn = RxnDfn()

        x = torch.rand(1,2,64,64)

        identity = rxn.id_conv(x)
        neighborhood = rxn.neighborhood_conv(x)
        update = rxn.update_universe(identity, neighborhood)
        next_x = rxn(x)

        sum_of_error = torch.abs(next_x - (x+rxn.dt*update)).sum()

        self.assertAlmostEqual(sum_of_error, 0.0)

        self.assertEqual(x.shape, next_x.shape)

    def test_id_conv(self):


        for channels in [2]:
            rxn = RxnDfn(internal_channels=channels, \
                    external_channels=channels)
            grid = torch.rand(1,channels,32,32)

            id_grid = rxn.id_conv(grid)

            sum_difference = (grid - id_grid).sum()

            self.assertAlmostEqual(0.0, sum_difference)


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


if __name__ == "__main__": #pragma: no cover

    unittest.main()
