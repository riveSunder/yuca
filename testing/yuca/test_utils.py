import unittest

import numpy as np
import torch

import matplotlib.pyplot as plt


from yuca.utils import query_kwargs, \
        seed_all, \
        get_mask, \
        get_bite_mask, \
        get_aperture, \
        prep_input, \
        make_target, \
        plot_grid_nbhd, \
        plot_kernel_growth

class TestQueryKwargs(unittest.TestCase):

    def setUp(self):
        pass 

    def test_query_kwargs(self):
        
        result_1 = query_kwargs("cat", "feline", cat="feline") 
        result_2 = query_kwargs("number_weekdays", 7, cat="feline") 
        result_3 = query_kwargs("number_weekdays", 7, number_weekdays=31) 

        self.assertEqual(result_1, "feline")
        self.assertEqual(result_2, 7)
        self.assertEqual(result_3, 31)

        for ii in range(100):

            default = np.random.randint(2)
            value = np.random.rand()

            result = query_kwargs("variable", default, variable=value)
            result_typo = query_kwargs("variable", default, varialbe=value)

            self.assertEqual(result, value)
            self.assertEqual(result_typo, default)
            self.assertNotEqual(result_typo, value)


class TestPlotKernelGrowth(unittest.TestCase):
    """

    def plot_grid_nbhd(grid, nbhd, update, my_cmap=plt.get_cmap("magma"), \
            titles=None, vmin=0, vmax=1):
    """ 
    def setUp(self):
        pass

    def test_plot_grid_nbhd(self):

        kernel = np.random.rand(32,32)
        growth_fn = lambda x: np.sin(x)
        
        fig = plot_kernel_growth(kernel, growth_fn)
        fig2 = plt.figure()

        self.assertEqual(type(fig2), type(fig))

class TestPlotGridNbhd(unittest.TestCase):
    """

    def plot_grid_nbhd(grid, nbhd, update, my_cmap=plt.get_cmap("magma"), \
            titles=None, vmin=0, vmax=1):
    """ 
    def setUp(self):
        pass

    def test_plot_grid_nbhd(self):

        grid = np.random.rand(32,32)
        nbhd = np.random.rand(32,32)
        update = np.random.rand(32,32)
        
        fig = plot_grid_nbhd(grid, nbhd, update)
        fig2 = plt.figure()

        self.assertEqual(type(fig2), type(fig))

class TestPrepInput(unittest.TestCase):

    def setUp(self):
        pass

    def test_prep_input(self):
        # minimal test
        img = np.random.rand(1,1,256,256)

        batch = prep_input(img)

        self.assertEqual(batch.shape[-1], img.shape[-1])

    def test_prep_input_batch_size(self):
        # test batch size arg
        
        img = np.random.rand(1,1,256,256)

        for batch_size in [1,4,8,32]:
            batch = prep_input(img, batch_size=batch_size)

            self.assertEqual(batch.shape[0], batch_size)

class TestMakeTarget(unittest.TestCase):

    def setUp(self):

        pass

    def test_make_target(self):
        # simple throughput test

        img = np.random.rand(1, 256,256)
        target = make_target(img)

        self.assertEqual(img.shape[-1], target.shape[-1])

class TestGetAperture(unittest.TestCase):

    def setUp(self):
        pass

    def test_aperture_full(self):

        img = np.random.rand(1,1, 256,256)
        aperture_radius = 0.98
        
        aperture = get_aperture(img, aperture_radius)

        self.assertEqual(img.shape, aperture.shape)
        self.assertEqual(1.0, aperture.mean())

    def test_aperture_shape(self):

        img = np.random.rand(1,1, 256,256)
        aperture_radius = np.clip(np.random.rand(),0.1,0.95)
        
        aperture = get_aperture(img, aperture_radius)

        self.assertEqual(img.shape, aperture.shape)

    def test_aperture_sum(self):

        for my_seed in [1234, 2345, 3456]: 

            seed_all(my_seed)
            img = np.random.rand(1,1, 256,256)
            aperture_radius = np.clip(np.random.rand(),0.1,0.95)
            
            aperture = get_aperture(img, aperture_radius)
            

            self.assertGreater(img.sum(), (img*aperture).sum())

    def test_aperture_center(self):

        for my_seed in [1234, 2345, 3456]: 

            seed_all(my_seed)
            img = np.random.rand(1,1, 256,256)
            aperture_radius = np.clip(np.random.rand(),0.1,0.95)
            
            aperture = get_aperture(img, aperture_radius)
            
            half_dim = img.shape[2] // 2

            self.assertEqual(1.0, aperture[0,0,half_dim,half_dim])

class TestGetMask(unittest.TestCase):

    def setUp(self):
        pass

    def test_mask_shape(self):

        img = np.random.rand(1,1, 256,256)
        
        mask = get_mask(img, radius=.1)

        self.assertEqual(img.shape, mask.shape)

    def test_mask_sum(self):

        for my_seed in [1234, 2345, 3456]: 

            seed_all(my_seed)
            img = np.random.rand(1,1, 256,256)
            
            mask = get_mask(img, radius=.5)

            self.assertGreater(img.sum(), (img*mask).sum())

    def test_mask_center(self):

        for my_seed in [1234, 2345, 3456]: 

            seed_all(my_seed)
            img = np.random.rand(1,1, 256,256)
            
            mask = get_mask(img, radius=.5)

            half_dim = img.shape[2] // 2

            self.assertEqual(0.0, mask[0,0,half_dim,half_dim])

class TestGetBiteMask(unittest.TestCase):

    def setUp(self):
        pass

    def test_bite_mask_shape(self):

        img = np.random.rand(1,1, 256,256)
        
        bite_mask = get_bite_mask(img, bite_radius=.1)

        self.assertEqual(img.shape, bite_mask.shape)

    def test_bite_mask_sum(self):

        for my_seed in [1234, 2345, 3456]: 

            seed_all(my_seed)
            img = np.random.rand(1,1, 256,256)
            
            bite_mask = get_bite_mask(img, bite_radius=.5)

            self.assertLess(img.sum(), bite_mask.sum())

            

class TestSeedAll(unittest.TestCase):

    def setUp(self):
        pass

    def test_seed_all_00(self):
        
        my_seeds = [13, 1337, 42, 12345, 100000]
        for seed in my_seeds:

            seed_all(seed)

            temp_a = np.random.randint(100)

            seed_all(seed)

            temp_b = np.random.randint(100)

            self.assertEqual(temp_a, temp_b)
    
    def test_seed_all_01(self):
        
        my_seeds = [13, 1337, 42, 12345, 100000]
        for seed in my_seeds:

            seed_all(seed)

            temp_a = np.random.rand(100)

            seed_all(seed)

            temp_b = np.random.rand(100)

            self.assertEqual(0.0, np.sum(temp_a - temp_b))

    def test_seed_all_02(self):
        
        my_seeds = [13, 1337, 42, 12345, 100000]
        for seed in my_seeds:

            seed_all(seed)

            temp_a = np.random.randn(100)

            seed_all(seed)

            temp_b = np.random.randn(100)

            self.assertEqual(0.0, np.sum(temp_a - temp_b))

    def test_seed_all_03(self):
        
        my_seeds = [13, 1337, 42, 12345, 100000]
        for seed in my_seeds:

            seed_all(seed)

            temp_aa = torch.randn(100)
            temp_ab = torch.rand(100)
            temp_ac = torch.randint(0, 100, size=(100,))

            seed_all(seed)

            temp_ba = torch.randn(100)
            temp_bb = torch.rand(100)
            temp_bc = torch.randint(0, 100, size=(100,))

            self.assertAlmostEqual(0.0, (temp_aa-temp_ba).sum())
            self.assertAlmostEqual(0.0, (temp_ab-temp_bb).sum())
            self.assertAlmostEqual(0.0, (temp_ac-temp_bc).sum())

if __name__ == "__main__":
    
    unittest.main(verbosity = 2)
