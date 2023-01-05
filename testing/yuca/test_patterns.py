import unittest

import numpy as np

from yuca.patterns import get_glider,\
        get_puffer,\
        get_pacman,\
        get_orbium, \
        get_geminium,\
        get_smooth_puffer,\
        get_smooth_glider

        
class TestPatterns(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_glider(self):

        test_glider = get_glider()
        self.assertEqual((8,8), test_glider.shape)
        self.assertEqual(5, np.sum(test_glider))

    def test_get_puffer(self):

        test_pattern = get_puffer()
        self.assertEqual((8,8), test_pattern.shape)
        self.assertEqual(16, np.sum(test_pattern))
        

    def test_get_pacman(self):

        test_pattern = get_pacman()
        #self.assertEqual((8,8), test_pattern.shape)
        #self.assertEqual(16, np.sum(test_pattern))

        self.assertEqual(np.ndarray, type(test_pattern))

    def test_get_orbium(self):

        test_pattern = get_orbium()
        #self.assertEqual((8,8), test_pattern.shape)
        #self.assertEqual(16, np.sum(test_pattern))

        self.assertEqual(np.ndarray, type(test_pattern))

    def test_get_geminium(self):

        test_pattern = get_geminium()
        #self.assertEqual((8,8), test_pattern.shape)
        #self.assertEqual(16, np.sum(test_pattern))

        self.assertEqual(np.ndarray, type(test_pattern))
        
    def test_get_smooth_puffer(self):

        test_pattern = get_smooth_puffer()
        #self.assertEqual((8,8), test_pattern.shape)
        #self.assertEqual(16, np.sum(test_pattern))

        self.assertEqual(np.ndarray, type(test_pattern))

    def test_get_smooth_glider(self):

        test_pattern = get_smooth_glider()
        #self.assertEqual((8,8), test_pattern.shape)
        #self.assertEqual(16, np.sum(test_pattern))

        self.assertEqual(np.ndarray, type(test_pattern))


if __name__ == "__main__": #pragma: no cover
    
    unittest.main()
