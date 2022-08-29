import unittest

import torch

from testing.yuca.test_multiverse import TestCA
from testing.yuca.test_activations import TestGaussian, TestDoGaussian
from testing.yuca.test_utils import TestQueryKwargs,\
        TestSeedAll, \
        TestGetMask, \
        TestGetBiteMask

if __name__ == "__main__":
    
    if not (torch.cuda.is_available()):
        msg = "\n   cuda not detected, tests will run on cpu only \n" 
        print(msg)
    unittest.main(verbosity=2)
