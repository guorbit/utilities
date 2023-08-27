import os
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

import numpy as np
import rasterio
import spectral
from PIL import Image
from scipy.ndimage import zoom


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


class RGBImageStrategyMultiThread:
    def __init__(
        self,
        image_path: str,
        image_size: tuple[int, int],
        image_resample=Image.Resampling.NEAREST,
        max_workers: int = 8,
    ):
        self.image_path = image_path
        self.image_filenames = np.array(
            sorted(os.listdir(self.image_path))
        )  #!update: added variable to initialiser
        self.image_size = image_size
        self.image_resample = image_resample
        self.max_workers = max_workers

    def __read_single_image_pil(self, filename, image_path, image_size, image_resample):
        image = Image.open(os.path.join(image_path, filename)).resize(
            image_size, image_resample
        )
        return np.array(image)

    def read_batch(self, batch_size: int, dataset_index: int) -> np.ndarray:
        batch_filenames = self.image_filenames[
            dataset_index : dataset_index + batch_size
        ]

        images = np.zeros((batch_size, self.image_size[0], self.image_size[1], 3))
        is_color = True

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(
                    self.__read_single_image_pil,
                    filename,
                    self.image_path,
                    self.image_size,
                    self.image_resample,
                ): i
                for i, filename in enumerate(batch_filenames)
            }
            for future in futures.as_completed(future_to_index):
                i = future_to_index[future]
                image = future.result()

                if len(image.shape) == 2 and is_color:
                    images = np.zeros(
                        (batch_size, self.image_size[0], self.image_size[1])
                    )
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


class HSImageStrategy:
    """
    Reads hyperspectral imagedata using Spectral Python
    """

    def __init__(
        self, image_path: str, image_size: tuple[int, int], package: Any = spectral
    ) -> None:
        self.image_path = image_path
        self.image_filenames = np.array(sorted(os.listdir(self.image_path)))
        self.image_size = image_size
        self.package = package
        self.bands = self.__get_channels()

    def __get_channels(self) -> int:
        # Open the first image to determine the number of channels
        first_image = self.package.open_image(
            os.path.join(self.image_path, self.image_filenames[0])
        )
        return first_image.shape[-1] if len(first_image.shape) == 3 else 1

    def read_batch(self, batch_size, dataset_index) -> np.ndarray:
        # read images with Spectral Python
        batch_filenames = self.image_filenames[
            dataset_index : dataset_index + batch_size
        ]

        images = np.zeros(
            (batch_size, self.image_size[0], self.image_size[1], self.bands)
        )
        is_color = True
        for i in range(batch_size):
            image = self.package.open_image(
                os.path.join(self.image_path, batch_filenames[i])
            )
            image_data = image.load()

            # Calculate the zoom factor for resizing
            zoom_factor = (
                self.image_size[0] / image_data.shape[0],
                self.image_size[1] / image_data.shape[1],
                1,
            )

            # Resize the image using scipy's zoom function
            resized_image = zoom(image_data, zoom_factor, order=1)

            if len(resized_image.shape) == 2 and is_color:
                images = np.zeros((batch_size, self.image_size[0], self.image_size[1]))
                is_color = False
            images[i, ...] = resized_image
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


class RasterImageStrategy:
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
            images[i, :, :, :] = np.resize(image, (self.bands, *self.image_size))

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


class RasterImageStrategyMultiThread:
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
        self, filename: str, package: Any, image_size: tuple[int, int, int]
    ) -> np.ndarray:
        with package.open(filename) as dataset:
            image = dataset.read()
        resized_image = np.resize(image, image_size)
        return resized_image

    def read_batch(self, batch_size: int, dataset_index: int) -> np.ndarray:
        batch_filenames = [
            os.path.join(self.image_path, filename)
            for filename in self.image_filenames[
                dataset_index : dataset_index + batch_size
            ]
        ]

        # Pre-allocate memory
        images = np.zeros(
            (batch_size, self.bands, self.image_size[0], self.image_size[1])
        )

        # Use ThreadPoolExecutor.map for more efficient multi-threading
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for i, image in enumerate(
                executor.map(
                    self.__read_single_image,
                    batch_filenames,
                    [self.package] * batch_size,
                    [(self.bands, *self.image_size)] * batch_size,
                )
            ):
                images[i, :, :, :] = image

        # Ensure channel-last orientation
        images = np.moveaxis(images, 1, 3)

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
