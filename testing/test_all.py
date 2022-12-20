import unittest

import torch

from testing.yuca.test_activations import TestGaussian, \
        TestDoGaussian,\
        TestGetSmoothStepsFn,\
        TestSmoothLifeKernel,\
        TestSmoothIntervals,\
        TestIdentity,\
        TestPolynomial,\
        TestCosOverX2,\
        TestGaussianMixture

from testing.yuca.test_clone import TestCloneFromCA
from testing.yuca.test_configs import TestConfigs
from testing.yuca.ca.test_continuous import TestCCA
from testing.yuca.test_kernels import TestGenericKernel,\
        TestGaussianKernel,\
        TestGaussianEdgeKernel,\
        TestDOGaussianKernel,\
        TestDOGaussianEdgeKernel,\
        TestLaplacianKernel
from testing.yuca.ca.test_neural import TestNCA
from testing.yuca.ca.test_reaction_diffusion import TestRxnDfn
from testing.yuca.test_utils import TestQueryKwargs,\
        TestSeedAll, \
        TestGetMask, \
        TestGetBiteMask,\
        TestGetAperture,\
        TestPrepInput, \
        TestMakeTarget, \
        TestPlotGridNbhd,\
        TestSaveFigSequence,\
        TestPlotKernelGrowth
from testing.yuca.test_patterns import TestPatterns

from testing.test_cli import TestCLI

if __name__ == "__main__": #pragma: no cover
    
    if not (torch.cuda.is_available()):
        msg = "\n   cuda not detected, tests will run on cpu only \n" 
        print(msg)
    unittest.main(verbosity=2)
