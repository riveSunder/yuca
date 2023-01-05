import unittest

from  yuca.wrappers.halting_wrapper import SimpleHaltingWrapper,\
        HaltingWrapper


class TestSimpleHaltingWrapper(unittest.TestCase):

    def setUp(self):
        self.wrapper = SimpleHaltingWrapper(ca_steps=4)
        self.wrapper.reset()

    def test_step(self):        
        
        action = self.wrapper.action_space.sample()

        obs, reward, done, info = self.wrapper.step(action)

        self.assertEqual(dict, type(info))   

class TestHaltingWrapper(TestSimpleHaltingWrapper):

    def setUp(self):
        self.wrapper = HaltingWrapper(ca_steps=4)
        self.wrapper.reset()


if __name__ == "__main__": #pragma: no cover
    unittest.main()
