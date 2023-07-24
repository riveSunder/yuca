import unittest

import numpy as np

from yuca.activations import Gaussian,\
        LaplacianOfGaussian,\
        DoGaussian,\
        GaussianMixture,\
        Identity,\
        Polynomial,\
        CosOverX2,\
        SmoothIntervals,\
        SmoothLifeKernel,\
        get_smooth_steps_fn

class TestGetSmoothStepsFn(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_smooth_steps_fn(self):
        
        ss_fn = get_smooth_steps_fn([[0.3,0.4]])

        x = np.arange(0,1.0,0.01)

        y = ss_fn(x)

        self.assertEqual(x.shape, y.shape)


class TestSmoothLifeKernel(unittest.TestCase):

    def setUp(self):
        self.activation = SmoothLifeKernel()

    def test_smooth_life_kernel(self):
        
        x = np.random.rand(32,32)

        y = self.activation(x)

        self.assertEqual(x.shape, y.shape)

class TestSmoothIntervals(unittest.TestCase):

    def setUp(self):
        self.activation = SmoothIntervals()

    def test_smooth_intervals(self):
        
        x = np.random.rand(1,32)

        y = self.activation(x)

        self.assertEqual(x.shape, y.shape)

class TestIdentity(unittest.TestCase):

    def setUp(self):
        self.activation = Identity()

    def test_identity(self):
        
        x = np.random.rand(1,32)

        y = self.activation(x)

        self.assertEqual(x.shape, y.shape)
        self.assertEqual(0.0, np.sum(x-y))

class TestPolynomial(unittest.TestCase):

    def setUp(self):
        self.activation = Polynomial()

    def test_polynomial(self):
        
        x = np.random.rand(1,32)

        y = self.activation(x)

        self.assertEqual(x.shape, y.shape)

class TestCosOverX2(unittest.TestCase):

    def setUp(self):
        self.activation = CosOverX2()

    def test_polynomial(self):
        
        x = np.random.rand(1,32)

        y = self.activation(x)

        self.assertEqual(x.shape, y.shape)

class TestLaplacianOfGaussian(unittest.TestCase):

    def setUp(self):

        pass

    def test_forward(self):

        my_log = LaplacianOfGaussian()

        for radius in np.arange(1,31):

            xx, yy = np.meshgrid(np.arange(-radius, radius + 1), \
                    np.arange(-radius, radius + 1))

            grid = np.sqrt(xx**2 + yy**2) / radius

            kernel = my_log(grid.reshape(1, 1, radius * 2 + 1, radius * 2 + 1))

            self.assertAlmostEqual(0.0, kernel.sum().item())
            self.assertEqual(radius*2+1, kernel.shape[-1])
            self.assertEqual(radius*2+1, kernel.shape[-2])
            self.assertEqual(1, 1)
            self.assertEqual(1, 1)

class TestGaussian(unittest.TestCase):
    """
    test for Gaussian 
    from yuca.activations
    """

    def setUp(self):

        pass

    def test_gaussian(self):

        sigma = 0.1
        for mu in [-1.0, -0.1, .2, 0.1, 0.0]:
            
            my_gaussian = Gaussian(mu=mu, sigma=sigma)
            max_gaussian = my_gaussian(mu)

            lesser_gaussian_1 = my_gaussian(mu - 0.1)
            lesser_gaussian_2 = my_gaussian(mu + 0.1)

            self.assertLess(lesser_gaussian_1, max_gaussian)
            self.assertLess(lesser_gaussian_2, max_gaussian)
            
class TestGaussianMixture(unittest.TestCase):
    """
    test for GaussianMixture 
    from yuca.activations
    """

    def setUp(self):

        pass

    def test_gaussian(self):

        for params in [[.1,0.1],[0.1, 0.01, .15, 0.015]]:
            for mode in [0, 1]:
            
                my_gaussian = GaussianMixture(mode=mode, parameters=params)
                x = np.arange(0,1.0,0.01)
                y = my_gaussian(x)

                self.assertEqual(x.shape, y.shape)

                if mode:
                    self.assertGreaterEqual(y.min(),-1.0)
                    self.assertLess(y.min(), 0.0)
                    self.assertLessEqual(y.max(), 1.0)
                else:
                    self.assertGreaterEqual(y.min(),0.0)
                    self.assertLessEqual(y.max(), 1.0)

class TestDoGaussian(unittest.TestCase):
    """
    test for DoGaussian (difference of Gaussians)
    from yuca.activations
    """

    def setUp(self):

        pass

    def test_dogaussian(self):

        sigma = 0.1

        for mu in [-1.0, -0.1, .2, 0.1, 0.0]:
            
            dog = DoGaussian(mu=mu, sigma=sigma)
            mu_value = dog(mu)

            lesser_gaussian_1 = dog(mu - 0.1)
            lesser_gaussian_2 = dog(mu + 0.1)

            self.assertLess(mu_value, lesser_gaussian_1)
            self.assertGreater(mu_value, lesser_gaussian_2)
            self.assertAlmostEqual(mu_value, 0.0)

            dog = DoGaussian(mu=mu, sigma=sigma, dx=-0.01)
            mu_value = dog(mu)

            lesser_gaussian_1 = dog(mu - 0.1)
            lesser_gaussian_2 = dog(mu + 0.1)

            self.assertGreater(mu_value, lesser_gaussian_1)
            self.assertLess(mu_value, lesser_gaussian_2)
            self.assertAlmostEqual(mu_value, 0.0)



if __name__ == "__main__": #pragma: no cover
    
    unittest.main()
