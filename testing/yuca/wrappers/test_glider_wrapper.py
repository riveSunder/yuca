import unittest

from  yuca.wrappers.glider_wrapper import GliderWrapper


class TestGliderWrapper(unittest.TestCase):

    def setUp(self):
        self.glider = GliderWrapper(ca_steps=4)

    def test_step(self):

        action = self.glider.action_space.sample()
        obs, reward, done, info = self.glider.step(action)

        self.assertEqual(dict, type(info))   


if __name__ == "__main__": #pragma: no cover
    unittest.main()
