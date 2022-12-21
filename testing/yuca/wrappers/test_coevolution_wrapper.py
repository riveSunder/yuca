import unittest

from  yuca.wrappers.coevolution_wrapper import CoevolutionWrapper


class TestCoevolutionWrapper(unittest.TestCase):

    def setUp(self):
        self.coevo = CoevolutionWrapper(ca_steps=4)

    def test_wrappers_step(self):        

        for ll in range(len(self.coevo.wrappers)):
            
            action = self.coevo.wrappers[ll].action_space.sample()

            obs, reward, done, info = self.coevo.wrappers[ll].step(action)

            self.assertEqual(dict, type(info))   

    def test_step(self):

        action = self.coevo.action_space.sample()
        obs, reward, done, info = self.coevo.step(action)

        self.assertEqual(dict, type(info))   


if __name__ == "__main__": #pragma: no cover
    unittest.main()
