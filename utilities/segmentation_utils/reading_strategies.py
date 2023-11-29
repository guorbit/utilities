import os
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
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
        Function loads a batch of image filenames starting from the given dataset index
        
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
        Calculates and returns the number of mini-batches that can be created from the \
        available image files from the target directory.

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
    :numpy.ndarray image_filenames: Array of image filenames obtained by sorting the list of files \
    """

    def __init__(
            self,
            image_path: str,
            image_size: tuple[int, int],
            image_resample: Image.Resampling = Image.Resampling.NEAREST,
    ):
        self.image_path = image_path
        self.image_filenames = np.array(
            sorted(os.listdir(self.image_path))
        )  # !update: added variable to initialiser
        self.image_size = image_size
        self.image_resample = image_resample

    def read_batch(self, batch_size: int, dataset_index: int) -> np.ndarray:
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
                          dataset_index: dataset_index + batch_size
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
        Calculates and returns the number of mini-batches that can be created from the \
        available image files from the target directory.

        Parameters
        ----------
        :int mini_batch: size of each mini batch

        Returns
        -------
        :return int: Number of mini-batches that can be formed from the given image filenames. \

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

    Parameters
    ----------
    :string image_path: path to the image directory
    :tuple image_size: specifies the dimensions of the input image (height, width)
    :int max_workers: number of threads to use for parallel processing

    Keyword Arguments
    -----------------
    :Image.Resampling image_resample: resampling method to use when resizing the image. \
    defaults to Image.Resampling.NEAREST (PIL)
    :numpy.ndarray image_filenames: Array of image filenames obtained by sorting the list of files \

    """

    def __init__(
            self,
            image_path: str,
            image_size: tuple[int, int],
            image_resample: Image.Resampling = Image.Resampling.NEAREST,
            max_workers: int = 8,
    ):
        self.image_path = image_path
        self.image_filenames = np.array(
            sorted(os.listdir(self.image_path))
        )  # !update: added variable to initialiser
        self.image_size = image_size
        self.image_resample = image_resample
        self.max_workers = max_workers

    def __read_single_image_pil(self, filename, image_path, image_size, image_resample):
        """
        Function to read a single image using PIL and resize it to the specified image size.

        Parameters:
        -----------
        :string filename: name of the image file
        :string image_path: path to the image directory
        :tuple image_size: specifies the dimensions of the input image (height, width)
        :Image.Resampling image_resample: resampling method to use when resizing the image. \
        defaults to Image.Resampling.NEAREST (PIL)

        Returns:
        --------
        :return np.ndarray[Any]: A batch of processed images that have been resized.


        """
        image = Image.open(os.path.join(image_path, filename)).resize(
            image_size, image_resample
        )
        return np.array(image)

    def read_batch(self, batch_size: int, dataset_index: int) -> np.ndarray:
        """
        Function loads a batch of image filenames starting from the given dataset index.

        Parameters
        ----------
        :int batch_size: the adjusted batch size to read
        :int dataset_index: specifies position of image batch within dataset

        Returns
        -------
        :return np.ndarray[Any]: A batch of processed images that have been resized and converted \
        to grayscale if needed.
        """
        batch_filenames = self.image_filenames[
                          dataset_index: dataset_index + batch_size
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
        """
        Calculates and returns the number of mini-batches that can be created from the \
        available image files from the target directory.

        Parameters
        ----------
        :int mini_batch: size of each mini batch

        Returns
        -------
        :return int: Number of mini-batches that can be formed from the given image filenames. \

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

        Parameters
        ----------
        :int seed: seed for random number generator to ensure reproducibility.

        """
        state = np.random.RandomState(seed)
        shuffled_indices = state.permutation(len(self.image_filenames))
        shuffled_indices = shuffled_indices.astype(int)
        self.image_filenames = self.image_filenames[shuffled_indices]


class HSImageStrategy:
    """
    Strategy optimized for reading hyperspectral images powered by backend OpenCV.

    Parameters
    ----------
    :string image_path: path to the image directory
    :tuple image_size: specifies the dimensions of the input image (height, width)
    :int bands: number of bands in the image

    Keyword Arguments
    -----------------
    :Any package: package to use for reading images. Defaults to OpenCV.
    :numpy.ndarray image_filenames: Array of image filenames obtained by sorting the list of files \
    """

    def __init__(
            self, image_path: str, image_size: tuple[int, int], package: Any = None
    ) -> None:
        self.image_path = image_path
        self.image_filenames = np.array(sorted(os.listdir(self.image_path)))
        self.image_size = image_size
        self.package = package or cv2
        self.bands = self.__get_channels()

    def __get_channels(self) -> int:
        """
        Function to determine the number of channels in the image.

        Returns
        -------
        :return int: number of channels in the image

        """
        # Open the first image to determine the number of channels
        sample_image_path = os.path.join(self.image_path, self.image_filenames[0])
        sample_image = self.package.imread(
            sample_image_path, self.package.IMREAD_UNCHANGED
        )
        return sample_image.shape[2] if len(sample_image.shape) == 3 else 1

    def read_batch(self, batch_size, dataset_index) -> np.ndarray:
        """
        Function loads a batch of image filenames starting from the given dataset index and \
        returns a batch of images.
        Each image is resized to the specified image size and converted to RGB if needed.

        Parameters
        ----------
        :int batch_size: the adjusted batch size to read
        :int dataset_index: specifies position of image batch within dataset
        
        Returns
        -------
        :return np.ndarray[Any]: A batch of processed images that have been resized and converted \
        to RGB if needed.

        """
        # Read a sample image to determine the number of bands

        # Initialize images array
        images = np.zeros(
            (batch_size, self.image_size[1], self.image_size[0], self.bands)
        )

        # Read images with OpenCV
        batch_filenames = self.image_filenames[
                          dataset_index: dataset_index + batch_size
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
        """
        Calculates and returns the number of mini-batches that can be created from the \
        available image files from the target directory.

        Parameters
        ----------
        :int mini_batch: size of each mini batch

        Returns
        -------
        :return int: Number of mini-batches that can be formed from the given image filenames. \

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

        Parameters
        ----------
        :int seed: seed for random number generator to ensure reproducibility.
        
        """
        state = np.random.RandomState(seed)
        shuffled_indices = state.permutation(len(self.image_filenames))
        shuffled_indices = shuffled_indices.astype(int)
        self.image_filenames = self.image_filenames[shuffled_indices]


class HSImageStrategyMultiThread:
    """
    Strategy optimized for reading hyperspectral images powered by backend OpenCV.
    Multi threaded version.

    Parameters
    ----------
    :string image_path: path to the image directory
    :tuple image_size: specifies the dimensions of the input image (height, width)
    :int bands: number of bands in the image
    :int max_workers: number of threads to use for parallel processing

    Keyword Arguments
    -----------------
    :Any package: package to use for reading images. Defaults to OpenCV.
    :numpy.ndarray image_filenames: Array of image filenames obtained by sorting the list of files \

    """

    def __init__(
            self,
            image_path: str,
            image_size: tuple[int, int],
            package: Any = None,
            max_workers: int = 8,
    ) -> None:
        self.image_path = image_path
        self.image_filenames = np.array(sorted(os.listdir(self.image_path)))
        self.image_size = image_size
        self.package = package or cv2
        self.bands = self.__get_channels()
        self.max_workers = max_workers

    def __get_channels(self) -> int:
        """
        Function to determine the number of channels in the image.

        Returns
        -------
        :return int: number of channels in the image

        """
        # Open the first image to determine the number of channels
        sample_image_path = os.path.join(self.image_path, self.image_filenames[0])
        sample_image = self.package.imread(
            sample_image_path, self.package.IMREAD_UNCHANGED
        )
        return sample_image.shape[2] if len(sample_image.shape) == 3 else 1

    def __read_single_image(
            self, filename: str, package: Any, image_size: tuple[int, int, int]
    ) -> np.ndarray:
        """
        Function to read a single image using OpenCV and resize it to the specified image size.

        Parameters:
        -----------
        :string filename: name of the image file
        :Any package: package to use for reading images. Defaults to OpenCV.
        :tuple image_size: specifies the dimensions of the input image (height, width)

        Returns:
        --------
        :return np.ndarray[Any]: A batch of processed images that have been resized.

        """
        image = package.imread(filename, package.IMREAD_UNCHANGED)
        image = package.resize(image, image_size)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = package.cvtColor(image, package.COLOR_BGR2RGB)
        return image

    def read_batch(self, batch_size, dataset_index) -> np.ndarray:
        """
        Function loads a batch of image filenames starting from the given dataset index and \
        returns a batch of images.
        Each image is resized to the specified image size and converted to RGB if needed.

        Parameters
        ----------
        :int batch_size: the adjusted batch size to read
        :int dataset_index: specifies position of image batch within dataset

        Returns
        -------
        :return np.ndarray[Any]: A batch of processed images that have been resized and converted \
        to RGB if needed.

        """
        # Initialize images array
        images = np.zeros(
            (batch_size, self.image_size[1], self.image_size[0], self.bands)
        )

        # Read images with OpenCV
        batch_filenames = self.image_filenames[
                          dataset_index: dataset_index + batch_size
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
        """
        Calculates and returns the number of mini-batches that can be created from the \
        available image files from the target directory.

        Parameters
        ----------
        :int mini_batch: size of each mini batch

        Returns
        -------
        :return int: Number of mini-batches that can be formed from the given image filenames. \

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

        """
        state = np.random.RandomState(seed)
        shuffled_indices = state.permutation(len(self.image_filenames))
        shuffled_indices = shuffled_indices.astype(int)
        self.image_filenames = self.image_filenames[shuffled_indices]


class RasterImageStrategy:
    """
    Strategy optimized for reading raster images powered by backend rasterio.

    Parameters
    ----------
    :string image_path: path to the image directory
    :tuple image_size: specifies the dimensions of the input image (height, width)

    Keyword Arguments
    -----------------
    :Image.Resampling image_resample: resampling method to use when resizing the image. \
    defaults to Image.Resampling.NEAREST (PIL)
    :Any package: package to use for reading images. Defaults to rasterio.
    :numpy.ndarray image_filenames: Array of image filenames obtained by sorting the list of files \
    """

    # read images with rasterio
    def __init__(
            self,
            image_path: str,
            image_size: tuple[int, int],
            image_resample=Image.Resampling.NEAREST,
            package: Any = None,
            bands_enabled: Union[list[bool], object] = None,
    ):
        self.image_path = image_path
        self.image_filenames = np.array(sorted(os.listdir(self.image_path)))
        self.image_size = image_size
        self.image_resample = image_resample
        self.package = package or rasterio
        # gets the number of bands for the dataset
        self.bands = package.open(
            os.path.join(self.image_path, self.image_filenames[0])
        ).count
        if bands_enabled is None:
            self.bands_enabled = [True] * self.bands
        else:
            self.bands_enabled = bands_enabled

    def read_batch(self, batch_size: int, dataset_index: int) -> np.ndarray:
        """
        Function loads a batch of image filenames starting from the given dataset index and \
        returns a batch of images.

        Parameters
        ----------
        :int batch_size: the adjusted batch size to read
        :int dataset_index: specifies position of image batch within dataset

        Returns
        -------
        :return np.ndarray[Any]: A batch of processed images that have been resized \
        
        """

        # read images with rasterio
        batch_filenames = self.image_filenames[
                          dataset_index: dataset_index + batch_size
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
        """
        Calculates and returns the number of mini-batches that can be created from the \
        available image files from the target directory.

        Parameters
        ----------
        :int mini_batch: size of each mini batch

        Returns
        -------
        :return int: Number of mini-batches that can be formed from the given image filenames. \

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

        Parameters
        ----------
        :int seed: seed for random number generator to ensure reproducibility.
    
        """
        state = np.random.RandomState(seed)
        shuffled_indices = state.permutation(len(self.image_filenames))
        shuffled_indices = shuffled_indices.astype(int)
        self.image_filenames = self.image_filenames[shuffled_indices]


class RasterImageStrategyMultiThread:
    """
    Strategy optimized for reading raster images powered by backend rasterio.
    Multi threaded version.

    Parameters
    ----------
    :string image_path: path to the image directory
    :tuple image_size: specifies the dimensions of the input image (height, width)
    :int max_workers: number of threads to use for parallel processing

    Keyword Arguments
    -----------------
    :Image.Resampling image_resample: resampling method to use when resizing the image. \
    defaults to Image.Resampling.NEAREST (PIL)
    :Any package: package to use for reading images. Defaults to rasterio.
    :numpy.ndarray image_filenames: Array of image filenames obtained by sorting the list of files \
    :int bands: number of bands in the image
    """

    # read images with rasterio
    def __init__(
            self,
            image_path: str,
            image_size: tuple[int, int],
            image_resample=Image.Resampling.NEAREST,
            max_workers: int = 8,

            bands_enabled: Union[list[bool], object] = None,
            package: Any = None,
    ):
        self.image_path = image_path
        self.image_filenames = np.array(sorted(os.listdir(self.image_path)))
        self.image_size = image_size
        self.image_resample = image_resample
        self.package = package or rasterio
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
        """
        Function to read a single image using rasterio and resize it to the specified image size.

        Parameters:
        -----------
        :string filename: name of the image file
        :Any package: package to use for reading images. Defaults to rasterio.  
        :tuple image_size: specifies the dimensions of the input image (height, width)

        Returns:
        --------
        :return np.ndarray[Any]: A batch of processed images that have been resized.

        """
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
        """
        Function loads a batch of image filenames starting from the given dataset index.

        Parameters
        ----------
        :int batch_size: the adjusted batch size to read
        :int dataset_index: specifies position of image batch within dataset

        Returns
        -------
        :return np.ndarray[Any]: A batch of processed images that have been resized \
        
        """
        batch_filenames = [
            os.path.join(self.image_path, filename)
            for filename in self.image_filenames[
                            dataset_index: dataset_index + batch_size
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
        """
        Calculates and returns the number of mini-batches that can be created from the \
        available image files from the target directory.

        Parameters
        ----------
        :int mini_batch: size of each mini batch

        Returns
        -------
        :return int: Number of mini-batches that can be formed from the given image filenames. 

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

        Parameters
        ----------
        :int seed: seed for random number generator to ensure reproducibility.

        """
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
        self.inner_idx = np.arange(self.ex_batch_size)

    def __resize_image_batch(self, image_batch, new_width, new_height):
        batch_size, old_height, old_width, _ = image_batch.shape

        # Create a set of indices for the new image
        x_indices = (np.arange(new_height) * (old_height / new_height)).astype(int)
        y_indices = (np.arange(new_width) * (old_width / new_width)).astype(int)

        # Use numpy's advanced indexing to pull out the correct pixels from the original image
        x_indices_mesh, y_indices_mesh = np.meshgrid(x_indices, y_indices, indexing='ij')

        # Repeat the indices arrays along the batch dimension
        x_indices_mesh = np.repeat(x_indices_mesh[np.newaxis, :, :], batch_size, axis=0)
        y_indices_mesh = np.repeat(y_indices_mesh[np.newaxis, :, :], batch_size, axis=0)

        # Index into the original image to get the resized images
        resized_images = image_batch[np.arange(batch_size)[:, np.newaxis, np.newaxis],
        x_indices_mesh, y_indices_mesh]

        return resized_images

    def read_batch(self, batch_size, dataset_index) -> np.ndarray:
        idx = dataset_index // self.ex_batch_size
        idx = self.dataset_idxs[idx]
        images = np.load(os.path.join(self.image_path, "batch_{}.npy".format(idx)))

        if self.bands_enabled is None:
            self.bands_enabled = [True] * images.shape[-1]
        images = images[:, :, :, self.bands_enabled]
        images = self.__resize_image_batch(images, self.image_size[0], self.image_size[1])
        if idx == self.last_batch_idx and images.shape[0] != batch_size:
            return images[:batch_size, ...]
        return images

    def get_dataset_size(self, mini_batch) -> int:
        return int(np.floor(self.dataset_size / float(mini_batch)))

    def get_image_size(self) -> tuple[int, int]:
        return self.image_size

    def shuffle_filenames(self, seed: int) -> None:
        state = np.random.RandomState(seed)
        remaining_idxs = self.dataset_idxs[:-1]
        # Shuffle the remaining batch indexes
        state.shuffle(remaining_idxs)
        # Append the last batch index back
        self.dataset_idxs = np.concatenate([remaining_idxs, [self.last_batch_idx]])
        state.shuffle(self.inner_idx)
