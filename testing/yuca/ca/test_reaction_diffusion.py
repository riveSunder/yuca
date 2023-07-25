import os
import unittest

import numpy as np
import torch

from yuca.ca.reaction_diffusion import RxnDfn

class TestRxnDfn(unittest.TestCase):

    def setUp(self):
        pass

    def test_grid_equilibrium(self):

        rxn = RxnDfn()

        grid = rxn.initialize_grid()

        self.assertAlmostEqual(0.42012, grid[:,0,:,:].mean().item(), 5)
        self.assertAlmostEqual(0.29253, grid[:,1,:,:].mean().item(), 5)

        my_batch_size = 10
        grid = rxn.initialize_grid(batch_size=my_batch_size)
        self.assertEqual(my_batch_size, grid.shape[0])
        self.assertAlmostEqual(0.42012, grid[:,0,:,:].mean().item(), 5)
        self.assertAlmostEqual(0.29253, grid[:,1,:,:].mean().item(), 5)

        my_dim = 16
        grid = rxn.initialize_grid(dim=my_dim)
        self.assertEqual(my_dim, grid.shape[-1])
        self.assertEqual(my_dim, grid.shape[-2])
        self.assertAlmostEqual(0.42012, grid[:,0,:,:].mean().item(), 5)
        self.assertAlmostEqual(0.29253, grid[:,1,:,:].mean().item(), 5)

        my_dim = (16,17)
        grid = rxn.initialize_grid(dim=my_dim)
        self.assertEqual(my_dim[0], grid.shape[-2])
        self.assertEqual(my_dim[1], grid.shape[-1])
        self.assertAlmostEqual(0.42012, grid[:,0,:,:].mean().item(), 5)
        self.assertAlmostEqual(0.29253, grid[:,1,:,:].mean().item(), 5)

    def test_forward(self):

        rxn = RxnDfn()

        x = torch.rand(1,2,64,64)

        next_x = rxn(x)

        self.assertEqual(x.shape, next_x.shape)


    def test_configs(self):

        rxndfn_1 = RxnDfn()
        rxndfn_1.no_grad()

        my_filepath = "./temp_delete_rxndfn_config.npy"

        my_config_1 = rxndfn_1.make_config()
        my_config_1_2 = rxndfn_1.make_config()

        rxndfn_1.save_config(my_filepath)
        rxndfn_1.restore_config(my_filepath)
        my_config_2 = rxndfn_1.make_config()

        rxndfn_2 = RxnDfn()
        rxndfn_2.no_grad()
        rxndfn_2.set_params(np.random.rand(*rxndfn_2.get_params().shape))
        my_config_3 = rxndfn_2.make_config()

        rxndfn_2.restore_config(my_filepath)
        my_config_4 = rxndfn_2.make_config()

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

    def test_update_universe(self):

        rxn = RxnDfn()

        x = torch.rand(1,2,64,64)

        identity = rxn.id_conv(x)
        neighborhood = rxn.neighborhood_conv(x)
        update = rxn.update_universe(identity, neighborhood)
        x_update = x+rxn.dt*update
        x_update_clamp = torch.clamp(x_update, 0, 1.0)

        next_x = rxn(x)


        sum_of_error = torch.abs(next_x - (x_update_clamp)).sum().numpy()

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
