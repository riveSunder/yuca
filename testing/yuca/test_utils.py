import unittest

import numpy as np
import torch

from yuca.utils import query_kwargs, seed_all

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
