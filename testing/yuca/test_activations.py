import unittest


from yuca.activations import Gaussian, DoGaussian

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



if __name__ == "__main__":
    
    unittest.main()
