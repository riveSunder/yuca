import os

import unittest

import numpy as np

import torch

from yuca.ca.continuous import CCA

class TestConfigs(unittest.TestCase):

    def setUp(self):
        pass

    def test_configs(self):

        file_path = os.path.abspath(__file__)

        testing_path = os.path.split(file_path)[0]
        root_path = os.path.split(testing_path)[0]

        config_directory = os.path.split(root_path)[0]
        config_directory = os.path.join(config_directory, "ca_configs")

        config_list = os.listdir(config_directory)

        for config_file in config_list:
            
            if config_file.endswith("npy"):
                try:
                    config_filepath = os.path.join(config_directory, config_file)
                    ca = CCA(ca_config=config_filepath)
                except:
                    message = f"could not load config {config_file}"
                    print(message)
                    self.assertFalse(True)

            

        self.assertTrue(True)


if __name__ == "__main__": #pragma: no cover

    unittest.main()
