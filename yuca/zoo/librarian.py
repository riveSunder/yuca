import os

import numpy as np

from yuca.utils import query_kwargs


class Librarian():

    def __init__(self, **kwargs):

        self.cur_path = os.path.split(os.path.realpath(__file__))[0]
        self.directory = query_kwargs("directory", self.cur_path, **kwargs) 

        default_config_dir = os.path.split(os.path.split(self.cur_path)[0])[0]
        default_config_dir = os.path.join(default_config_dir, "ca_configs")
        self.config_directory = query_kwargs(\
                "config_directory", default_config_dir, **kwargs)

        self.verbose = query_kwargs("verbose", True, **kwargs)

        self.update_index()

    def update_index(self):
        """
        update the list of (string) names of patterns in the zoo directory
        """

        pattern_names = os.listdir(self.directory)

        remove_list = []
        for elem in pattern_names:
            if ".py" in elem \
                    or ".ipynb" in elem \
                    or "csv" in elem \
                    or "__pycache__" in elem:
                remove_list.append(elem)
                
        for elem in remove_list:
            pattern_names.remove(elem)

        pattern_names = [os.path.splitext(elem)[0] for elem in pattern_names]

        pattern_names.sort()

        self.index = pattern_names

    def store(self, pattern: np.array, pattern_name: str = "my_pattern",\
            config_name: str = "unspecified", entry_point="not specified",\
            commit_hash="not_specified"):

        counter = 0
        file_path = os.path.join(self.directory, f"{pattern_name}{counter:03}.npy")

        while os.path.exists(file_path):
            counter += 1
            file_path = os.path.join(self.directory, f"{pattern_name}{counter:03}.npy")

            if counter >= 1000:
                # shouldn't be here, assuming less than 1000 patterns of same name
                print(f"more than {counter} variants of pattern {pattern_name}, "\
                        f"consider choosing a new name.")

        meta_path = os.path.join(self.directory, 
                f"{pattern_name}{counter:03}.csv")


        if config_name == "unspecified" and self.verbose:
            print(f"warning, no config supplied for {pattern_name}")

        with open(meta_path, "w") as f:
            f.write(f"ca_config,{config_name}")
            f.write(f"\ncommit_hash,{commit_hash}")
            f.write(f"\nentry_point,{entry_point}")

        np.save(file_path, pattern) 

        if self.verbose:
            print(f"pattern {pattern_name} saved to {file_path}")
            print(f"pattern {pattern_name} metadata saved to {meta_path}")

        self.index.append(f"{pattern_name}  {counter:03}")


    def load(self, pattern_name: str) -> tuple([np.array, str]):
        """
        load pattern from disk
        """

        file_path = os.path.join(self.directory, f"{pattern_name}.npy")
        meta_path = os.path.join(self.directory, f"{pattern_name}.csv")


        pattern = np.load(file_path)

        with open(meta_path) as f:
            metadata = f.readlines()

            ca_config = metadata[0].split(",")[1]

            try:
                entry_point = metadata[3].split(",")[1]
            except:
                entry_point = "none"
            try:
                commit_hash = metadata[1].split(",")[1]
            except:
                commit_hash = "none"

        if self.verbose:
            print(f"pattern {pattern_name} loaded from {file_path}")
            print(f"pattern {pattern_name} metadata loaded from {meta_path}")


        return pattern, ca_config, entry_point, commit_hash


    def crop(self, pattern: np.array, row: tuple, column: tuple) -> np.array:
        """
        crop a pattern to row[0], column[0] to row[1], column[1]

        cropping is applied to the last 2 dimensions
        """
        new_pattern = 1.0 * pattern[..., row[0]:row[1], column[0]:column[1]]

        return new_pattern

    def autocrop(self, pattern: np.array) -> np.array:
        """
        attempt to automatically crop a pattern to an active bounding box
        """
        pass

if __name__ == "__main__":

    librarian = Librarian()
    
