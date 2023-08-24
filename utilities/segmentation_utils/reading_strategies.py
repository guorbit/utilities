import os
from types import ModuleType
from typing import Protocol, Type, Union

import numpy as np
import rasterio
from PIL import Image


class MockRasterio:
    def __init__(self, n , size, bands, dtypes):
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
        return np.zeros((self.bands,self.size[0],self.size[1]), self.dtypes[0])

    # these functions are invoked when a 'with' statement is executed
    def __enter__(self):
        # called at the beginning of a 'with' block
        return self  # returns instance of MockRasterio class itself

    def __exit__(self, type, value, traceback):
        # called at the end of a 'with' block
        pass


class IReader(Protocol):
    def read_batch(self, start: int, end: int) -> None:
        ...

    def get_dataset_size(self) -> None:
        ...


class RGBImageStrategy:
    def __init__(
        self,
        image_path: str,
        image_size: tuple[int, int],
        image_resample=Image.Resampling.NEAREST,
    ):
        self.image_path = image_path
        self.image_filenames = np.array(
            sorted(os.listdir(self.image_path))
        )  #!update: added variable to initialiser
        self.image_size = image_size
        self.image_resample = image_resample

    def read_batch(self, batch_size, dataset_index) -> np.ndarray:
        # read images with PIL

        for i in range(batch_size):
            image_index = i + dataset_index
            image = Image.open(
                os.path.join(self.image_path, self.image_filenames[image_index])
            ).resize(self.image_size, self.image_resample)
            image = np.array(image)
            image = image / 255
        return image

    def get_dataset_size(self, mini_batch) -> int:
        dataset_size = int(np.floor(len(self.image_filenames) / float(mini_batch)))
        return dataset_size


class HyperspectralImageStrategy:
    # read images with rasterio
    def __init__(
        self,
        image_path: str,
        image_resize: tuple[int, int],
        image_resample=Image.Resampling.NEAREST,
        package: Union[MockRasterio, ModuleType] = rasterio,
    ):
        self.image_path = image_path
        self.image_filenames = np.array(sorted(os.listdir(self.image_path)))
        self.image_resize = image_resize
        self.image_resample = image_resample
        self.package = package
        # gets the number of bands for the dataset
        self.bands = package.open(
            os.path.join(self.image_path, self.image_filenames[0])
        ).count
        print("-----------My very cool bands--------: ",self.bands)

    def read_batch(self, batch_size:int, dataset_index:int) -> np.ndarray:
        # read images with rasterio
        batch_filenames = self.image_filenames[
            dataset_index : dataset_index + batch_size
        ]

        # defines the array that will contain the images
        images = np.zeros(
            (batch_size, self.bands, self.image_resize[0], self.image_resize[1])
        )
        for i, filename in enumerate(batch_filenames):
            with self.package.open(os.path.join(self.image_path, filename)) as dataset:
                # .read() returns a numpy array that contains the raster cell values in your file.
                image = dataset.read()
            images[i, :, :, :] = np.resize(image, self.image_resize)

        # ensures channel-last orientation for the reader
        images = np.moveaxis(images, 1, 3)

        return np.array(images)

    def get_dataset_size(self, mini_batch) -> int:
        dataset_size = int(np.floor(len(self.image_filenames) / float(mini_batch)))
        return dataset_size
