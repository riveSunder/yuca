import unittest

import torch

from testing.yuca.ca.test_cca import TestCCA
from testing.yuca.ca.test_nca import TestNCA
from testing.yuca.ca.test_reaction_diffusion import TestRxnDfn
from testing.yuca.test_activations import TestGaussian, TestDoGaussian
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
from testing.yuca.test_configs import TestConfigs
from testing.yuca.test_kernels import TestGenericKernel,\
        TestGaussianKernel,\
        TestGaussianEdgeKernel,\
        TestDOGaussianKernel,\
        TestDOGaussianEdgeKernel,\
        TestLaplacianKernel
from testing.yuca.test_clone import TestCloneFromCA

if __name__ == "__main__": #pragma: no cover
    
    if not (torch.cuda.is_available()):
        msg = "\n   cuda not detected, tests will run on cpu only \n" 
        print(msg)
    unittest.main(verbosity=2)
