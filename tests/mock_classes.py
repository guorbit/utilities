import numpy as np


class MockRasterio:
    def __init__(self, n, size, bands, dtypes):
        self.n = n
        self.size = size
        self.bands = bands
        self.dtypes = dtypes

    def open(self, *args, **kwargs):
        return self

    @property
    def count(self) -> int:
        return self.bands

    def read(self, *args, **kwargs):
        return np.zeros((self.bands, self.size[0], self.size[1]), self.dtypes[0])

    # these functions are invoked when a 'with' statement is executed
    def __enter__(self):
        # called at the beginning of a 'with' block
        return self  # returns instance of MockRasterio class itself

    def __exit__(self, type, value, traceback):
        # called at the end of a 'with' block
        pass