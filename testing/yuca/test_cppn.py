import unittest

from yuca.cppn import CPPN, CPPNPlus
from yuca.utils import seed_all
import torch

class TestCPPN(unittest.TestCase):
    
    def setUp(self):
        seed_all(my_seed=13)

        self.cppn = CPPN()

    def test_get_action(self):
        
        action_1 = self.cppn.get_action()
        action_2 = self.cppn.get_action(self.cppn.grid)

        random_grid = torch.rand_like(self.cppn.grid) 

        action_3 = self.cppn.get_action(random_grid)

        self.assertAlmostEqual(0.0, (action_1-action_2).sum())
        self.assertNotEqual(0.0, (action_1-action_3).sum())


    def test_use_grad(self):

        self.assertFalse(self.cppn.use_grad)


class TestCPPNPlus(TestCPPN):
    
    def setUp(self):
        seed_all(my_seed=13)

        self.cppn = CPPNPlus()

    def test_get_action(self):
        
        action_1 = self.cppn.get_pattern_action()
        action_2 = self.cppn.get_pattern_action(self.cppn.grid)

        random_grid = torch.rand_like(self.cppn.grid) 

        action_3 = self.cppn.get_pattern_action(random_grid)

        self.assertAlmostEqual(0.0, (action_1-action_2).sum())
        self.assertNotEqual(0.0, (action_1-action_3).sum())
    
    def test_get_rule_action(self):

       rule_action = self.cppn.get_rule_action() 

       self.assertEqual(rule_action.shape[0], self.cppn.params_agent.num_params)

if __name__ == "__main__":

    unittest.main()
