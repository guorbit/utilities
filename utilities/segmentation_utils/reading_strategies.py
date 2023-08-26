import os
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

import numpy as np
import rasterio
from PIL import Image


class IReader(Protocol):
    def read_batch(self, batch_size: int, dataset_index: int) -> np.ndarray:
        ...

    def get_dataset_size(self, minibatch: int) -> int:
        ...

    def get_image_size(self) -> tuple[int, int]:
        ...

    def shuffle_filenames(self, seed: int) -> None:
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
        batch_filenames = self.image_filenames[
            dataset_index : dataset_index + batch_size
        ]

        images = np.zeros((batch_size, self.image_size[0], self.image_size[1], 3))
        is_color = True
        for i in range(batch_size):
            image = Image.open(
                os.path.join(self.image_path, batch_filenames[i])
            ).resize(self.image_size, self.image_resample)
            image = np.array(image)
            if len(image.shape) == 2 and is_color:
                images = np.zeros((batch_size, self.image_size[0], self.image_size[1]))
                is_color = False
            images[i, ...] = image
        return images

    def get_dataset_size(self, mini_batch) -> int:
        dataset_size = int(np.floor(len(self.image_filenames) / float(mini_batch)))
        return dataset_size

    def get_image_size(self) -> tuple[int, int]:
        return self.image_size

    def shuffle_filenames(self, seed: int) -> None:
        state = np.random.RandomState(seed)
        shuffled_indices = state.permutation(len(self.image_filenames))
        shuffled_indices = shuffled_indices.astype(int)
        self.image_filenames = self.image_filenames[shuffled_indices]


class HyperspectralImageStrategy:
    # read images with rasterio
    def __init__(
        self,
        image_path: str,
        image_size: tuple[int, int],
        image_resample=Image.Resampling.NEAREST,
        package: Any = rasterio,
    ):
        self.image_path = image_path
        self.image_filenames = np.array(sorted(os.listdir(self.image_path)))
        self.image_size = image_size
        self.image_resample = image_resample
        self.package = package
        # gets the number of bands for the dataset
        self.bands = package.open(
            os.path.join(self.image_path, self.image_filenames[0])
        ).count

    def read_batch(self, batch_size: int, dataset_index: int) -> np.ndarray:
        # read images with rasterio
        batch_filenames = self.image_filenames[
            dataset_index : dataset_index + batch_size
        ]

        # defines the array that will contain the images
        images = np.zeros(
            (batch_size, self.bands, self.image_size[0], self.image_size[1])
        )
        for i, filename in enumerate(batch_filenames):
            with self.package.open(os.path.join(self.image_path, filename)) as dataset:
                # .read() returns a numpy array that contains the raster cell values in your file.
                image = dataset.read()
            images[i, :, :, :] = np.resize(image, self.image_size)

        # ensures channel-last orientation for the reader
        images = np.moveaxis(images, 1, 3)

        return np.array(images)

    def get_dataset_size(self, mini_batch) -> int:
        dataset_size = int(np.floor(len(self.image_filenames) / float(mini_batch)))
        return dataset_size

    def get_image_size(self) -> tuple[int, int]:
        return self.image_size

    def shuffle_filenames(self, seed: int) -> None:
        state = np.random.RandomState(seed)
        shuffled_indices = state.permutation(len(self.image_filenames))
        shuffled_indices = shuffled_indices.astype(int)
        self.image_filenames = self.image_filenames[shuffled_indices]


class HyperspectralImageStrategyMultiThread:
    # read images with rasterio
    def __init__(
        self,
        image_path: str,
        image_size: tuple[int, int],
        image_resample=Image.Resampling.NEAREST,
        max_workers: int = 8,
        package: Any = rasterio,
    ):
        self.image_path = image_path
        self.image_filenames = np.array(sorted(os.listdir(self.image_path)))
        self.image_size = image_size
        self.image_resample = image_resample
        self.package = package
        self.max_workers = max_workers
        # gets the number of bands for the dataset
        self.bands = package.open(
            os.path.join(self.image_path, self.image_filenames[0])
        ).count

    def __read_single_image(
        self, filename: str, image_path: str, package: Any, image_size: tuple[int, int]
    ):
        with package.open(os.path.join(image_path, filename)) as dataset:
            image = dataset.read()
        resized_image = np.resize(image, image_size)
        return resized_image

    def read_batch(self, batch_size: int, dataset_index: int) -> np.ndarray:
        batch_filenames = self.image_filenames[
            dataset_index : dataset_index + batch_size
        ]

        # defines the array that will contain the images
        images = np.zeros(
            (batch_size, self.bands, self.image_size[0], self.image_size[1])
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(
                    self.__read_single_image,
                    filename,
                    self.image_path,
                    self.package,
                    self.image_size,
                ): i
                for i, filename in enumerate(batch_filenames)
            }
            for future in futures.as_completed(future_to_index):
                i = future_to_index[future]
                images[i, :, :, :] = future.result()

        # ensures channel-last orientation for the reader
        images = np.moveaxis(images, 1, 3)

        return np.array(images)

    def get_dataset_size(self, mini_batch) -> int:
        dataset_size = int(np.floor(len(self.image_filenames) / float(mini_batch)))
        return dataset_size

    def get_image_size(self) -> tuple[int, int]:
        return self.image_size

    def shuffle_filenames(self, seed: int) -> None:
        state = np.random.RandomState(seed)
        shuffled_indices = state.permutation(len(self.image_filenames))
        shuffled_indices = shuffled_indices.astype(int)
        self.image_filenames = self.image_filenames[shuffled_indices]
