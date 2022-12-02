import os

import torch

import unittest

from yuca.clone import clone_from_ca

from yuca.ca.neural import NCA


class TestCloneFromCA(unittest.TestCase):

    def setUp(self):
        pass

    def test_clone_from_ca(self):

        kwargs = {"max_steps": 2,\
                "save_name": "test_nca.pt",
                "hidden_channels": 16,\
                "max_hidden_channels": 16,
                "max_iterations": 2}


        ca_config = "orbium.npy"


        file_directory = os.path.abspath(__file__).split("/")
        root_directory = os.path.join(*file_directory[:-3])
        log_directory = os.path.join("/", root_directory, "logs")

        save_path = os.path.join(log_directory, kwargs["save_name"])
        os.system(f"rm {save_path}")

        clone_from_ca(ca_config=ca_config, **kwargs)

        self.assertTrue(os.path.exists(save_path))

        nca = NCA(hidden_channels=kwargs["max_hidden_channels"]) 

        nca.load_state_dict(torch.load(save_path, map_location = torch.device("cpu")))

        self.assertTrue(True)
        
        os.system(f"rm {save_path}")

if __name__ == "__main__":

    unittest.main()
