"""
FlowGenerator is a wrapper around the keras ImageDataGenerator class.
"""

import math
import os
from typing import Optional

import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from PIL import Image
from tqdm import tqdm

from utilities.segmentation_utils import ImagePreprocessor
from utilities.segmentation_utils.constants import ImageOrdering


class FlowGenerator:
    """
    Initializes the flow generator object,
    which can be used to read in images for semantic segmentation.
    Additionally, the reader can apply augmentation on the images,
    and one-hot encode them on the fly.

    Note: in case the output is a column vector it has to be in the shape (x, 1)

    Parameters
    ----------
    :string image: path to the image directory
    :string mask: path to the mask directory
    :int batch_size: batch size
    :tuple image_size: image size
    :tuple output_size: output size


    :int num_classes: number of classes

    Keyword Arguments
    -----------------
    :bool shuffle: whether to shuffle the dataset or not
    :int batch_size: batch size
    :bool preprocessing_enabled: whether to apply preprocessing or not
    :int seed: seed for flow from directory
    :int preprocessing_seed: seed for preprocessing, defaults to None

    Raises
    ------
    :ValueError: if the output size is not a tuple of length 2
    :ValueError: if the output size is not a square matrix or a column vector
    """

    preprocessing_seed = None
    preprocessing_queue_image = None
    preprocessing_queue_mask = None

    def __init__(
        self,
        image_path: str,
        mask_path: str,
        image_size: tuple[int, int],
        output_size: tuple[int, int],
        num_classes: int,
        shuffle: bool = True,
        batch_size: int = 2,
        preprocessing_enabled: bool = True,
        seed: int = 909,
        preprocessing_seed: Optional[int] = None,
    ):
        if len(output_size) != 2:
            raise ValueError("The output size has to be a tuple of length 2")
        if output_size[1] != 1 and output_size[0] != output_size[1]:
            raise ValueError(
                "The output size has to be a square matrix or a column vector"
            )

        self.image_path = image_path
        self.mask_path = mask_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.output_size = output_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.seed = seed
        self.preprocessing_enabled = preprocessing_enabled
        self.preprocessing_seed = preprocessing_seed
        self.__make_generator()
        print("Reading images from: ", self.image_path)

    def get_dataset_size(self) -> int:
        """
        Returns the length of the dataset

        Returns
        -------
        :returns int: length of the dataset
        """

        return len(os.listdir(os.path.join(self.image_path, "img")))

    def set_preprocessing_pipeline(
        self,
        preprocessing_queue_image: ImagePreprocessor.IPreprocessor,
        preprocessing_queue_mask: ImagePreprocessor.IPreprocessor,
    ) -> None:
        """
        Sets the preprocessing pipeline

        Parameters
        ----------
        :PreprocessingQueue preprocessing_queue_image: preprocessing queue for images
        :PreprocessingQueue preprocessing_queue_mask: preprocessing queue for masks
        """
        self.preprocessing_queue_image = preprocessing_queue_image
        self.preprocessing_queue_mask = preprocessing_queue_mask

    def __make_generator(self):
        """
        Creates the generator
        """

        image_datagen = ImageDataGenerator()
        mask_datagen = ImageDataGenerator()

        if self.output_size[1] == 1:
            # only enters if the output is a column vector
            # such no need to define it otherwise
            dimension = math.sqrt(self.output_size[0])
            self.output_reshape = (int(dimension), int(dimension))

        image_generator = image_datagen.flow_from_directory(
            self.image_path,
            class_mode=None,  # type: ignore
            seed=self.seed,
            batch_size=self.batch_size,
            target_size=self.image_size,
        )

        mask_generator = mask_datagen.flow_from_directory(
            self.mask_path,
            class_mode=None,  # type: ignore
            seed=self.seed,
            batch_size=self.batch_size,
            target_size=self.output_size,
            color_mode="grayscale",
        )
        if (
            self.preprocessing_queue_image is None
            and self.preprocessing_queue_mask is None
        ):
            #!Possibly in the wrong place as it has to be regenerated every time
            (
                self.preprocessing_queue_image,
                self.preprocessing_queue_mask,
            ) = ImagePreprocessor.generate_default_queue()
        elif (
            self.preprocessing_queue_image is None
            or self.preprocessing_queue_mask is None
        ):
            raise ValueError("Both queues must be passed or none")

        self.train_generator = zip(image_generator, mask_generator)
        self.train_generator = self.preprocess(self.train_generator)

    def get_generator(self):
        """
        Returns the generator object

        Returns
        -------
        :return ImageDataGenerator: generator object
        """
        return self.train_generator

    def preprocess(self, generator_zip):
        """
        Preprocessor function encapsulates both the image, and mask generator objects.
        Augments the images and masks and onehot encodes the masks

        Parameters
        ----------
        :tuple generator_zip: tuple of image and mask generator
        :int, optional state: random state for reproducibility, defaults to None

        Returns
        -------
        :return tuple(tf.Tensor,tf.Tensor): generator batch of image and mask
        """
        for img, mask in generator_zip:
            if self.preprocessing_enabled:
                for i_image, i_mask in zip(img, mask):
                    # random state for reproducibility
                    if self.preprocessing_seed is None:
                        image_seed = np.random.randint(0, 100000)
                    else:
                        state = np.random.RandomState(self.preprocessing_seed)
                        image_seed = state.randint(0, 100000)

                    i_image, i_mask = ImagePreprocessor.augmentation_pipeline(
                        image=i_image,
                        mask=i_mask,
                        input_size=self.image_size,
                        output_size=self.output_size,
                        output_reshape=self.output_reshape,
                        seed=image_seed,
                        #!both preprocessing queues are assigned by this time
                        image_queue=self.preprocessing_queue_image,  # type: ignore
                        mask_queue=self.preprocessing_queue_mask,  # type: ignore
                    )
            mask = ImagePreprocessor.onehot_encode(
                mask, self.output_size, self.num_classes
            )
            yield (img, mask)


class FlowGeneratorExperimental(Sequence):
    """
    Initializes the flow generator object,
    which can be used to read in images for semantic segmentation.
    Additionally, the reader can apply augmentation on the images,
    and one-hot encode them on the fly.
    
    Note: in case the output is a column vector it has to be in the shape (x, 1)
    Note: this is an experimental version of the flow generator, which uses a \
    custom implemented dataloader instead of the keras ImageDataGenerator
    #TODO: Instead of using direct paths, and arguments, reading heads should be used
    #TODO: as it reduces the number of arguments, and makes the code more readable and reduces
    #TODO: cupling

    Parameters
    ----------
    :string image: path to the image directory
    :string mask: path to the mask directory
    :int batch_size: 
    :tuple image_size: specifies the size of the input image
    :tuple output_size: specifies the size of the output mask
    :list[bool] channel_mask: specifies which channels of the input image to use
    :int num_classes: number of classes in the output mask


    Keyword Arguments
    -----------------
    :bool, optional shuffle: whether to shuffle the dataset or not, defaults to True
    :int batch_size: specifies the number of images read in one batch, defaults to 2
    :bool preprocessing_enabled: whether to apply preprocessing or not, defaults to True
    :int seed: seed for flow from directory
    :int preprocessing_seed: seed for preprocessing, defaults to None
    :preprocessing_queue_image: preprocessing queue for images
    :preprocessing_queue_mask: preprocessing queue for masks
    :bool read_weights: whether to read the weights from the mask directory, defaults to False
    :string weights_path: path to the weights directory, defaults to None
    :int shuffle_counter: the seed offset used for shuffling, defaults to 0
    :ImageOrdering image_ordering: the ordering of the image channels, defaults to channels_last
    
    Raises
    ------
    :ValueError: if the names of the images and masks do not match
    :ValueError: if the output size is not a tuple of length 2
    :ValueError: if the output size is not a square matrix or a column vector
    """

    def __init__(
        self,
        image_path: str,
        mask_path: str,
        image_size: tuple[int, int],
        output_size: tuple[int, int],
        channel_mask: list[bool],
        num_classes: int,
        shuffle: bool = True,
        batch_size: int = 2,
        preprocessing_enabled: bool = True,
        seed: int = 909,
        preprocessing_seed: Optional[int] = None,
        preprocessing_queue_image: ImagePreprocessor.IPreprocessor = ImagePreprocessor.generate_image_queue(),
        preprocessing_queue_mask: ImagePreprocessor.IPreprocessor = ImagePreprocessor.generate_mask_queue(),
        read_weights: bool = False,
        weights_path: Optional[str] = None,
        shuffle_counter: int = 0,
        image_ordering: ImageOrdering = ImageOrdering.CHANNEL_LAST,
    ):
        if len(output_size) != 2:
            raise ValueError("The output size has to be a tuple of length 2")
        if output_size[1] != 1 and output_size[0] != output_size[1]:
            raise ValueError(
                "The output size has to be a square matrix or a column vector"
            )

        self.image_path = image_path
        self.mask_path = mask_path
        self.batch_size = batch_size
        self.mini_batch = batch_size
        self.image_size = image_size
        self.output_size = output_size
        self.channel_mask = np.array(channel_mask)
        self.n_channels = np.sum(channel_mask)
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.seed = seed
        self.preprocessing_enabled = preprocessing_enabled
        self.preprocessing_seed = preprocessing_seed
        self.read_weights = read_weights
        self.weights_path = weights_path
        self.preprocessing_queue_image = preprocessing_queue_image
        self.preprocessing_queue_mask = preprocessing_queue_mask
        self.shuffle_counter = shuffle_counter
        self.image_ordering = image_ordering

        self.image_filenames = np.array(sorted(os.listdir(self.image_path)))
        self.mask_filenames = np.array(sorted(os.listdir(self.mask_path)))

        # should be moved out as a strategy
        if self.read_weights:
            weights_df = pd.read_csv(self.weights_path, header=None)
            weights_np = weights_df.to_numpy()
            print(weights_np.shape)
            # sort the numpy array by the first column
            weights_np = weights_np[weights_np[:, 0].argsort()]

            print(weights_np)
            self.weights = weights_np[:, 1:].astype(np.float64)
            weight_names = weights_np[:, 0]
            for mask, weight_name in zip(self.mask_filenames, weight_names):
                if mask != weight_name:
                    raise ValueError("The mask and weight directories do not match")

        self.linked_data = [self.image_filenames, self.mask_filenames]
        if self.read_weights:
            self.linked_data.append(self.weights)
        
        self.__shuffle_filenames()
        self.dataset_size = self.__len__()

        print("Validating dataset...")
        for i_name, m_name in tqdm(zip(self.image_filenames, self.mask_filenames)):
            if i_name != m_name:
                raise ValueError("The image and mask directories do not match")

        self.image_batch_store = None
        self.mask_batch_store = None
        self.validity_index = 0

        if self.output_size[1] == 1:
            # only enters if the output is a column vector
            # such no need to define it otherwise
            dimension = math.sqrt(self.output_size[0])
            self.output_reshape = (int(dimension), int(dimension))
            self.column_vector = True
        else:
            self.output_reshape = self.output_size
            self.column_vector = False

        print("Reading images from: ", self.image_path)

    def set_preprocessing_pipeline(
        self,
        preprocessing_queue_image: ImagePreprocessor.IPreprocessor,
        preprocessing_queue_mask: ImagePreprocessor.IPreprocessor,
    ) -> None:
        """
        Sets the preprocessing pipeline

        Parameters
        ----------
        :PreprocessingQueue preprocessing_queue_image: preprocessing queue for images
        :PreprocessingQueue preprocessing_queue_mask: preprocessing queue for masks
        """
        self.preprocessing_queue_image = preprocessing_queue_image
        self.preprocessing_queue_mask = preprocessing_queue_mask

    def set_mini_batch_size(self, batch_size: int) -> None:
        """
        Function to set the appropriate minibatch size. Required to allign batch size in the \
        reader with the model. Does not change the batch size of the reader.

        Parameters
        ----------
        :int batch_size: the mini batch size

        Raises
        ------
        :raises ValueError: if the mini batch size is larger than the batch size
        :raises ValueError: if the batch size is not divisible by the mini batch size
        """
        if batch_size > self.batch_size:
            raise ValueError("The mini batch size cannot be larger than the batch size")
        if self.batch_size % batch_size != 0:
            raise ValueError("The batch size must be divisible by the mini batch size")
        self.mini_batch = batch_size

    def __read_batch(self, start: int, end: int) -> None:
        # read image batch
        batch_image_filenames = self.image_filenames[start:end]
        batch_mask_filenames = self.mask_filenames[start:end]
        for image, mask in zip(batch_image_filenames, batch_mask_filenames):
            if image != mask:
                raise ValueError("The image and mask directories do not match")

        # calculate number of mini batches in a batch
        n = self.batch_size // self.mini_batch

        batch_images = np.zeros(
            (
                n,
                self.mini_batch,
                self.image_size[0],
                self.image_size[1],
                self.n_channels,
            )
        )

        batch_masks = np.zeros(
            (
                n,
                self.mini_batch,
                self.output_reshape[0],
                self.output_reshape[1],
                self.num_classes,
            )
        )

        # preprocess and assign images and masks to the batch
        for i in range(n):
            raw_masks = np.zeros(
                (self.mini_batch, self.output_reshape[0], self.output_reshape[1])
            )

            for j in range(self.mini_batch):
                image_index = i * self.mini_batch + j
            
                image = Image.open(
                    os.path.join(self.image_path, batch_image_filenames[image_index])
                ).resize(self.image_size, Image.ANTIALIAS)
                
                image = np.array(image)


                mask = Image.open(
                    os.path.join(self.mask_path, batch_mask_filenames[image_index])
                ).resize(self.output_reshape)

                mask = np.array(mask)
                # image = image[:, :, self.channel_mask]

                if self.preprocessing_enabled:
                    if self.preprocessing_seed is None:
                        image_seed = np.random.randint(0, 100000)
                    else:
                        state = np.random.RandomState(self.preprocessing_seed)
                        image_seed = state.randint(0, 100000)

                    (
                        image,
                        mask,
                    ) = ImagePreprocessor.augmentation_pipeline(
                        image,
                        mask=mask,
                        seed=image_seed,
                        #!both preprocessing queues are assigned by this time
                        image_queue=self.preprocessing_queue_image,  # type: ignore
                        mask_queue=self.preprocessing_queue_mask,  # type: ignore
                    )

                batch_images[i, j, :, :, :] = image
                # NOTE: this provides the flexibility required to process both
                # column and matrix vectors
                raw_masks[j, :, :] = mask

            batch_masks[i, :, : , :] = ImagePreprocessor.onehot_encode(
                raw_masks, self.num_classes
            )

        # chaches the batch
        self.image_batch_store = batch_images
        self.mask_batch_store = batch_masks

        # required to check when to read the next batch

    def __len__(self) -> int:
        return int(np.floor(len(self.image_filenames) / float(self.mini_batch)))

    def __getitem__(self, index) -> tuple[np.ndarray, np.ndarray]:
        # check if the batch is already cached
        index = index % self.dataset_size

        if index < self.validity_index - self.batch_size // self.mini_batch:
            self.validity_index = 0

        if index == self.validity_index:
            self.__read_batch(index * self.batch_size, (index + 1) * self.batch_size)
            self.validity_index = (self.batch_size // self.mini_batch) + index

        # slices new batch
        store_index = (self.batch_size // self.mini_batch) - (
            self.validity_index - index
        )

        batch_images = self.image_batch_store[store_index, ...]  # type: ignore
        batch_masks = self.mask_batch_store[store_index, ...]  # type: ignore
        if self.column_vector:
            batch_masks = np.reshape(
                batch_masks,
                (
                    self.mini_batch,
                    batch_masks.shape[1] * batch_masks[2],
                    self.num_classes,
                ),
            )

        if self.image_ordering == ImageOrdering.CHANNEL_FIRST:
            batch_images = np.moveaxis(batch_images, -1, 1)
            batch_masks = np.moveaxis(batch_masks, -1, 1)

        if self.read_weights:
            batch_weights = self.weights[
                index * self.batch_size : (index + 1) * self.batch_size, ...
            ]

            return batch_images, batch_masks, batch_weights
        else:
            return batch_images, batch_masks

    def on_epoch_end(self) -> None:
        # Shuffle image and mask filenames
        self.__shuffle_filenames()

    def __shuffle_filenames(self) -> None:
        if self.shuffle:
            state = np.random.RandomState(self.seed + self.shuffle_counter)
            self.shuffle_counter += 1
            shuffled_indices = state.permutation(len(self.image_filenames))
            shuffled_indices = shuffled_indices.astype(int)
            for array in self.linked_data:
                array = array[shuffled_indices]
