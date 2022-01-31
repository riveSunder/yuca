import unittest

import numpy as np

from yuca.utils import query_kwargs

class TestQueryKwargs(unittest.TestCase):

    def setUp(self):
        pass 

    def test_query_kwargs(self):
        
        result_1 = query_kwargs("cat", "feline", cat="feline") 
        result_2 = query_kwargs("number_weekdays", 7, cat="feline") 
        result_3 = query_kwargs("number_weekdays", 7, number_weekdays=31) 

        self.assertEqual(result_1, "feline")
        self.assertEqual(result_2, 7)
        self.assertEqual(result_3, 31)

        for ii in range(100):

            default = np.random.randint(2)
            value = np.random.rand()

            result = query_kwargs("variable", default, variable=value)
            result_typo = query_kwargs("variable", default, varialbe=value)

            self.assertEqual(result, value)
            self.assertEqual(result_typo, default)
            self.assertNotEqual(result_typo, value)

if __name__ == "__main__":
    
    unittest.main(verbosity = 2)
