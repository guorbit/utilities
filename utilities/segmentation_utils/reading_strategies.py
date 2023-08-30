import os
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol, Union

import cv2
import numpy as np
import pandas as pd
import rasterio
from PIL import Image


class IReader(Protocol):
    """
    Interface meant to be implemented by all reading strategies for usage in \
    flowgenerator dataset reader.

    """
    def read_batch(self, batch_size: int, dataset_index: int) -> np.ndarray:
        """
        Function loads a batch of image filenames starting from the given dataset index and \
        
        Parameters
        ----------
        :int batch_size: the adjusted batch size to read
        :int dataset_index: specifies position of image batch within dataset

        Returns
        -------
        :return np.ndarray[Any]: A batch of processed images that have been resized \
        """
        ...

    def get_dataset_size(self, minibatch: int) -> int:
        """
        Calculates and returns the number of mini-batches that can be created from the available image \
        files from the target directory.

        Parameters
        ----------
        :int minibatch: size of each mini batch
        """
        ...

    def get_image_size(self) -> tuple[int, int]:
        """
        Returns the dimensions (height, width) of the images, as a tuple of integers.

        Returns
        -------
        :return tuple[int, int]: Dimensions of the images as a tuple of integers.
        """
        ...

    def shuffle_filenames(self, seed: int) -> None:
        """
        Shuffle the order of image filenames using the provided seed.
        
        Parameters
        ----------
        :int seed: seed for random number generator to ensure reproducibility.
        """
        ...


class RGBImageStrategy:
    """
    Strategy optimized for reading RGB images powered by backend PIL.

    Parameters
    ----------
    :string image_path: path to the image directory
    :tuple image_size: specifies the dimensions of the input image (height, width)
    
    Keyword Arguments
    -----------------
    :Image.Resampling image_resample: resampling method to use when resizing the image. \
    defaults to Image.Resampling.NEAREST (PIL)
    :return numpy.ndarray: Array of image filenames obtained by sorting the list of files \
    in the specified image path.
    """

    def __init__(
        self,
        image_path: str,
        image_size: tuple[int, int],
        image_resample:Image.Resampling=Image.Resampling.NEAREST,
    ):
        self.image_path = image_path
        self.image_filenames = np.array(
            sorted(os.listdir(self.image_path))
        )  #!update: added variable to initialiser
        self.image_size = image_size
        self.image_resample = image_resample

    def read_batch(self, batch_size:int, dataset_index:int) -> np.ndarray:
        """
        Function loads a batch of image filenames starting from the given dataset index and \
        returns a batch of images.
        Each image is resized to the specified image size and converted to grayscale provided.

        Parameters
        ----------
        :int batch_size: the adjusted batch size to read
        :int dataset_index: specifies position of image batch within dataset
        
        Returns
        -------
        :return np.ndarray[Any]: A batch of processed images that have been resized and converted \
        to grayscale if needed.
        """
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
        """
        Calculates and returns the number of mini-batches that can be created from the available image \
        files from the target directory.

        Parameters
        ----------
        :int mini_batch: size of each mini batch

        Returns
        -------
        :return int: Number of mini-batches that can be formed from the given image filenames. \
        Result is rounded down to the nearest integer using np.floor to ensure all available \
        images are included.

        """
        dataset_size = int(np.floor(len(self.image_filenames) / float(mini_batch)))
        return dataset_size

    def get_image_size(self) -> tuple[int, int]:
        """
        Returns the dimensions (height, width) of the images, as a tuple of integers.

        Returns
        -------
        :return tuple[int, int]: Dimensions of the images as a tuple of integers.


        """
        return self.image_size

    def shuffle_filenames(self, seed: int) -> None:
        """
        Shuffle the order of image filenames using the provided seed.
        The order of filenames is rearranged based on these shuffled indices,\
         creating a new order for the filenames.

        Parameters
        ----------
        :int seed: seed for random number generator to ensure reproducibility.

        """

        state = np.random.RandomState(seed)
        shuffled_indices = state.permutation(len(self.image_filenames))
        shuffled_indices = shuffled_indices.astype(int)
        self.image_filenames = self.image_filenames[shuffled_indices]


class RGBImageStrategyMultiThread:
    """
    Strategy optimized for reading RGB images powered by backend PIL.
    Multi threaded version.

    parameters:

    keyword arguments: 

    """

    def __init__(
        self,
        image_path: str,
        image_size: tuple[int, int],
        image_resample:Image.Resampling=Image.Resampling.NEAREST,
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
        """

        """
        return self.image_size

    def shuffle_filenames(self, seed: int) -> None:
        state = np.random.RandomState(seed)
        shuffled_indices = state.permutation(len(self.image_filenames))
        shuffled_indices = shuffled_indices.astype(int)
        self.image_filenames = self.image_filenames[shuffled_indices]


class HSImageStrategy:
    """
    Strategy optimized for reading hyperspectral images powered by backend OpenCV.

    """

    def __init__(
        self, image_path: str, image_size: tuple[int, int], package: Any = cv2
    ) -> None:
        self.image_path = image_path
        self.image_filenames = np.array(sorted(os.listdir(self.image_path)))
        self.image_size = image_size
        self.package = package
        self.bands = self.__get_channels()

    def __get_channels(self) -> int:
        # Open the first image to determine the number of channels
        sample_image_path = os.path.join(self.image_path, self.image_filenames[0])
        sample_image = self.package.imread(
            sample_image_path, self.package.IMREAD_UNCHANGED
        )
        return sample_image.shape[2] if len(sample_image.shape) == 3 else 1

    def read_batch(self, batch_size, dataset_index) -> np.ndarray:
        # Read a sample image to determine the number of bands

        # Initialize images array
        images = np.zeros(
            (batch_size, self.image_size[1], self.image_size[0], self.bands)
        )

        # Read images with OpenCV
        batch_filenames = self.image_filenames[
            dataset_index : dataset_index + batch_size
        ]

        for i in range(batch_size):
            image_path = os.path.join(self.image_path, batch_filenames[i])
            image = self.package.imread(image_path, self.package.IMREAD_UNCHANGED)

            # Resize the image
            image = self.package.resize(image, self.image_size)

            # If the image is color, convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = self.package.cvtColor(image, self.package.COLOR_BGR2RGB)

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


class HSImageStrategyMultiThread:
    def __init__(
        self,
        image_path: str,
        image_size: tuple[int, int],
        package: Any = cv2,
        max_workers: int = 8,
    ) -> None:
        self.image_path = image_path
        self.image_filenames = np.array(sorted(os.listdir(self.image_path)))
        self.image_size = image_size
        self.package = package
        self.bands = self.__get_channels()
        self.max_workers = max_workers

    def __get_channels(self) -> int:
        # Open the first image to determine the number of channels
        sample_image_path = os.path.join(self.image_path, self.image_filenames[0])
        sample_image = self.package.imread(
            sample_image_path, self.package.IMREAD_UNCHANGED
        )
        return sample_image.shape[2] if len(sample_image.shape) == 3 else 1

    def __read_single_image(
        self, filename: str, package: Any, image_size: tuple[int, int, int]
    ) -> np.ndarray:
        image = package.imread(filename, package.IMREAD_UNCHANGED)
        image = package.resize(image, image_size)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = package.cvtColor(image, package.COLOR_BGR2RGB)
        return image

    def read_batch(self, batch_size, dataset_index) -> np.ndarray:
        # Initialize images array
        images = np.zeros(
            (batch_size, self.image_size[1], self.image_size[0], self.bands)
        )

        # Read images with OpenCV
        batch_filenames = self.image_filenames[
            dataset_index : dataset_index + batch_size
        ]

        image_paths = [
            os.path.join(self.image_path, batch_filenames[i]) for i in range(batch_size)
        ]

        with ThreadPoolExecutor() as executor:
            results = executor.map(
                self.__read_single_image,
                image_paths,
                [self.package] * batch_size,
                [self.image_size] * batch_size,
            )

        for i, image in enumerate(results):
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


class RasterImageStrategy:
    """
    Strategy optimized for reading raster images powered by backend rasterio.
    """

    # read images with rasterio
    def __init__(
        self,
        image_path: str,
        image_size: tuple[int, int],
        image_resample=Image.Resampling.NEAREST,
        package: Any = rasterio,
        bands_enabled: Union[list[bool], object] = None,
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
        if bands_enabled is None:
            self.bands_enabled = [True] * self.bands
        else:
            self.bands_enabled = bands_enabled

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
                for j in range(self.bands):
                    if not self.bands_enabled[j]:
                        continue
                    image = dataset.read(j + 1)
                    image = np.resize(image, self.image_size)
                    images[i, j, :, :] = image
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
    """
    Strategy optimized for reading raster images powered by backend rasterio.
    Multi threaded version.
    """

    # read images with rasterio
    def __init__(
        self,
        image_path: str,
        image_size: tuple[int, int],
        image_resample=Image.Resampling.NEAREST,
        max_workers: int = 8,
        package: Any = rasterio,
        bands_enabled: Union[list[bool], object] = None,
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
        if bands_enabled is None:
            self.bands_enabled = [True] * self.bands
        else:
            self.bands_enabled = np.array(bands_enabled)

        self.n_enabled = sum(self.bands_enabled)

    def __read_single_image(
        self, filename: str, package: Any, image_size: tuple[int, int, int]
    ) -> np.ndarray:
        image = np.zeros((self.bands, *image_size[1:None]))
        with package.open(filename) as dataset:
            for j in range(self.bands):
                if not self.bands_enabled[j]:
                    continue
                band = dataset.read(j + 1)
                band = np.resize(band, image_size[1:None])
                image[j, :, :] = band

        return image[self.bands_enabled, :, :]

    def read_batch(self, batch_size: int, dataset_index: int) -> np.ndarray:
        batch_filenames = [
            os.path.join(self.image_path, filename)
            for filename in self.image_filenames[
                dataset_index : dataset_index + batch_size
            ]
        ]

        # Pre-allocate memory
        images = np.zeros(
            (batch_size, self.n_enabled, self.image_size[0], self.image_size[1])
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


class BatchReaderStrategy:
    def __init__(
        self,
        image_path: str,
        image_size: tuple[int, int],
        package: Any = np,
        bands_enabled: Union[list[bool], object] = None,
    ) -> None:
        self.image_path = image_path
        meta_path = os.path.abspath(os.path.join(image_path, os.pardir))

        # Read the info.csv file containing number of images and batch size the data was processed at
        df = pd.read_csv(os.path.join(meta_path, "info.csv"), index_col=0)
        # Read the first row for n_image

        n_image = df.iloc[0][0]

        # Read the second row for batch_size
        batch_size = df.iloc[1][0]

        self.ex_batch_size = batch_size
        self.dataset_size = n_image
        # last batch of the dataset
        self.last_batch_idx = n_image // batch_size
        self.dataset_idxs = np.arange(self.last_batch_idx + 1)

        self.image_size = image_size
        self.package = package

        self.bands_enabled = bands_enabled

    def read_batch(self, batch_size, dataset_index) -> np.ndarray:
        idx = dataset_index // self.ex_batch_size
        images = np.load(os.path.join(self.image_path, "batch_{}.npy".format(idx)))

        if self.bands_enabled is None:
            self.bands_enabled = [True] * images.shape[-1]
        images = images[:, :, :, self.bands_enabled]

        if idx == self.last_batch_idx and images.shape[0] != batch_size:
            print("last irregular batch")
            return images[:batch_size, ...]
        return images

    def get_dataset_size(self, mini_batch) -> int:
        return int(np.floor(self.dataset_size / float(mini_batch)))

    def get_image_size(self) -> tuple[int, int]:
        return self.image_size

    def shuffle_filenames(self, seed: int) -> None:
        state = np.random.RandomState(seed)
        state.shuffle(self.dataset_idxs)
