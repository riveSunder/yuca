import os

import unittest

import numpy as np

import torch

import yuca.kernels
from yuca.kernels import get_laplacian_kernel


class TestLaplacianKernel(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_laplacian_kernel(self):

        laplacian = get_laplacian_kernel(radius=1)

        target = torch.tensor([[[[0,1.,0],[1.,-4.,1.],[0,1.,0]]]])

        self.assertNotIn(False, target == laplacian)
        self.assertTrue(type(laplacian) == torch.Tensor)
        

class TestGenericKernel(unittest.TestCase):

    def setUp(self):
        self.kernel_function = yuca.kernels.get_cosx2_kernel

    def test_kernel(self):

        for radius in [1,3,7,13,27]:
            kernel = self.kernel_function(radius=radius)

            self.assertTrue(kernel.shape[-1], radius * 2 + 1)
            self.assertTrue(kernel.shape[-2], kernel.shape[-1])


class TestGaussianKernel(TestGenericKernel):

    def setUp(self):
        self.kernel_function = yuca.kernels.get_gaussian_kernel


class TestDOGaussianKernel(TestGenericKernel):

    def setUp(self):
        self.kernel_function = yuca.kernels.get_dogaussian_kernel

class TestGaussianEdgeKernel(TestGenericKernel):

    def setUp(self):
        self.kernel_function = yuca.kernels.get_gaussian_edge_kernel

class TestDOGaussianEdgeKernel(TestGenericKernel):

    def setUp(self):
        self.kernel_function = yuca.kernels.get_dogaussian_edge_kernel

if __name__ == "__main__":

    unittest.main()
