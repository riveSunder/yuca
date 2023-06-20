import subprocess
import os

import unittest

class TestCLI(unittest.TestCase):

    def setUp(self):
        pass

    def test_evolve(self):

        for ca_type in ("NCA", "CCA", "RxnDfn"):
            evolve_pattern_cmd = f"python -m yuca.evolve -b 1 "\
                    f"-c 1 -d cpu -g 1 -k 13 "\
                    f"-l 3 -m 38 -p 8 -s 42 "\
                    f"-t my_test_tag -ca {ca_type} -hc 16"
                    
            if ca_type == "CCA":
                evolve_pattern_cmd += f" -cc orbium.npy"

            output = subprocess.check_output(evolve_pattern_cmd.split(" "))

            output = str(output)

            self.assertIn("evolution completed successfully", output)

            cleanup_logs_cmd = "rm -rf logs/*my_test_tag*"
            cleanup_gif_cmd = "rm -rf assets/*my_test_tag*"
            cleanup_configs_cmd = "rm -rf ca_configs/*my_test_tag*"

            os.system(cleanup_logs_cmd)
            os.system(cleanup_gif_cmd)
            os.system(cleanup_configs_cmd)

    def test_clone(self):

        # default args for clone  

        for max_steps in [1, 2]:
            for iterations in [1,2]:
                clone_cmd = f"python -m yuca.clone -e 0.1 -i {iterations} -m {max_steps} -o test.pt"

                clone_output = str(subprocess.check_output(clone_cmd.split(" ")))


                self.assertIn("saving to ", clone_output)

        cleanup_cmd = "rm logs/test.pt"
        os.system(cleanup_cmd)

    def test_plot(self):
        
        plot_cmd = "python -m yuca.plot -s 0 -i logs/exp_test_tag_1671048884_seed13.npy"

        plot_output = str(subprocess.check_output(plot_cmd.split(" ")))
        self.assertIn("plot finished", plot_output)
        
if __name__ == "__main__": #pragma: no cover

    unittest.main()
