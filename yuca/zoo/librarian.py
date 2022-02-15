import os

import numpy as np

from yuca.utils import query_kwargs


class Librarian():

    def __init__(self, **kwargs):

        self.cur_path = os.path.split(os.path.realpath(__file__))[0]
        self.directory = query_kwargs("directory", self.cur_path, **kwargs) 
        self.verbose = query_kwargs("verbose", True, **kwargs)

        self.update_index()

    def update_index(self):
        """
        update the list of (string) names of patterns in the zoo directory
        """

        pattern_names = os.listdir(self.directory)

        for elem in pattern_names:
            if ".py" in elem:
                pattern_names.remove(elem)

        pattern_names = [os.path.splitext(elem)[0] for elem in pattern_names]

        self.index = pattern_names

    def store(self, pattern: np.array, pattern_name: str = "my_pattern"):

        counter = 0
        file_path = os.path.join(self.directory, f"{pattern_name}{counter:03}.npy")

        while os.path.exists(file_path):
            counter += 1
            file_path = os.path.join(self.directory, f"{pattern_name}.npy")

            if counter > 1000:
                # shouldn't be here, assuming less than 1000 patterns of same name
                print(f"more than {counter} variants of pattern {pattern_name}, "\
                        f"consider choosing a new name.")

        np.save(file_path, pattern) 

        if self.verbose:
            print(f"pattern {pattern_name} saved to {file_path}")


    def load(self, pattern_name: str):
        """
        load pattern from disk
        """
        pass

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
    
