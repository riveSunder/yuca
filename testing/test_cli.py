import subprocess
import os

import unittest

class TestCLI(unittest.TestCase):

    def setUp(self):
        pass

    def test_evolve(self):

        evolve_pattern_cmd = f"python -m yuca.evolve -b 1 "\
                f"-c 1 -d cpu -g 1 -k 13 "\
                f"-l 3 -m 38 -p 8 -s 42 "\
                f"-t my_test_tag -cc orbium.npy"

        output = subprocess.check_output(evolve_pattern_cmd.split(" "))

        output = str(output)

        self.assertIn("evolution completed successfully", output)

        cleanup_logs_cmd = "rm -rf logs/*my_test_tag*"
        cleanup_gif_cmd = "rm -rf assets/*my_test_tag*"
        cleanup_configs_cmd = "rm -rf ca_configs/*my_test_tag*"

        os.system(cleanup_logs_cmd)
        os.system(cleanup_gif_cmd)
        os.system(cleanup_configs_cmd)

if __name__ == "__main__":

    unittest.main()
