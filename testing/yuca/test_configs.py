import os

import unittest

import numpy as np

import torch

from yuca.ca.continuous import CCA
from yuca.configs import get_smooth_life_config,\
    get_orbium_config,\
    get_geminium_config


class TestConfigs(unittest.TestCase):

    def setUp(self):
        pass

    def test_configs(self):

        file_path = os.path.abspath(__file__)

        this_file_path = os.path.split(file_path)[0]
        testing_path = os.path.split(this_file_path)[0]

        config_directory = os.path.split(testing_path)[0]
        config_directory = os.path.join(config_directory, "ca_configs")

        config_list = os.listdir(config_directory)
        temp_config_path = os.path.join(testing_path, "temp_config.npy")

        for config_file in config_list:
            
            if config_file.endswith("npy"):
                config_file += "\n"
                try:
                    config_filepath = os.path.join(config_directory, config_file)
                    ca = CCA(ca_config=config_filepath)

                    temp_config = ca.make_config()
                    ca.save_config(temp_config_path)
                except:
                    message = f"could not load config {config_file}"
                    print(message)
                    self.assertFalse(True)

            os.system(f"rm {temp_config_path}")
            

        self.assertTrue(True)

    def test_config_functions(self):
        
        config_smoothlife = get_smooth_life_config()
        orbium_config = get_orbium_config()
        geminium_config = get_geminium_config()

        ca = CCA()

        for my_config in [config_smoothlife, orbium_config, geminium_config]:
            ca.load_config(my_config)

if __name__ == "__main__": #pragma: no cover

    unittest.main()
