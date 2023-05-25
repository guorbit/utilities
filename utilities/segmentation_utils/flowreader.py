"""
FlowGenerator is a wrapper around the keras ImageDataGenerator class.
"""

import math
import os
from typing import Optional

import numpy as np
import tensorflow as tf
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence

from utilities.segmentation_utils import ImagePreprocessor


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
        preprocessing_queue_image: ImagePreprocessor.PreprocessorInterface,
        preprocessing_queue_mask: ImagePreprocessor.PreprocessorInterface,
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
        channel_mask: list[bool],
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

        self.image_filenames = os.listdir(os.path.join(self.image_path))
        self.mask_filenames = os.listdir(os.path.join(self.mask_path))

        self.image_batch_store = np.zeros(
            (1, self.batch_size, image_size[0], image_size[1], self.n_channels)
        )
        self.mask_batch_store = np.zeros((1, self.batch_size, 1, 1, num_classes))
        self.validity_index = 0

        if self.output_size[1] == 1:
            # only enters if the output is a column vector
            # such no need to define it otherwise
            dimension = math.sqrt(self.output_size[0])
            self.output_reshape = (int(dimension), int(dimension))
        else:
            self.output_reshape = self.output_size

        print("Reading images from: ", self.image_path)

    def set_preprocessing_pipeline(
        self,
        preprocessing_queue_image: ImagePreprocessor.PreprocessorInterface,
        preprocessing_queue_mask: ImagePreprocessor.PreprocessorInterface,
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
        Function to set the appropriate minibatch size. Required to allign batch size in the reader with the model.\
        Does not change the batch size of the reader.

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

    def read_batch(self, start: int, end: int) -> None:
        # read image batch
        batch_image_filenames = self.image_filenames[start:end]
        batch_mask_filenames = batch_image_filenames
        tf.print(batch_image_filenames)
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
        if self.output_size[1] == 1:
            column = True
            batch_masks = np.zeros(
                (n, self.mini_batch, self.output_size[0], self.num_classes)
            )
        else:
            column = False
            batch_masks = np.zeros(
                (
                    n,
                    self.mini_batch,
                    self.output_size[0],
                    self.output_size[1],
                    self.num_classes,
                )
            )

        # preprocess and assign images and masks to the batch
        for i in range(n):
            raw_masks = np.zeros(
                (self.mini_batch, self.output_size[0] * self.output_size[1], 1)
            )
            for j in range(self.mini_batch):
                image = Image.open(
                    os.path.join(self.image_path, batch_image_filenames[j])
                ).resize(self.image_size, Image.ANTIALIAS)

                image = np.array(image)
                image = image / 255

                mask = Image.open(
                    os.path.join(self.mask_path, batch_mask_filenames[j])
                ).resize(self.output_reshape)
                
                mask = np.array(mask)
                image = image[:, :, self.channel_mask]

                batch_images[i, j, :, :, :] = image

                if self.preprocessing_enabled:
                    if self.preprocessing_seed is None:
                        image_seed = np.random.randint(0, 100000)
                    else:
                        state = np.random.RandomState(self.preprocessing_seed)
                        image_seed = state.randint(0, 100000)

                    (
                        batch_images[i, j, :, :, :],
                        mask,
                    ) = ImagePreprocessor.augmentation_pipeline(
                        image=batch_images[i, j, :, :, :],
                        mask=mask,
                        input_size=self.image_size,
                        output_size=self.output_size,
                        output_reshape=self.output_reshape,
                        seed=image_seed,
                        #!both preprocessing queues are assigned by this time
                        image_queue=self.preprocessing_queue_image,  # type: ignore
                        mask_queue=self.preprocessing_queue_mask,  # type: ignore
                    )
                
                mask = np.reshape(mask, self.output_size)

                raw_masks[j, :, :] = mask

            batch_masks[i, :, :, :] = ImagePreprocessor.onehot_encode(
                raw_masks, self.output_size, self.num_classes
            )

        # chaches the batch
        self.image_batch_store = batch_images
        self.mask_batch_store = batch_masks

        # required to check when to read the next batch

    def __len__(self):
        return int(np.floor(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, index) -> tuple[np.ndarray, np.ndarray]:
        # check if the batch is already cached
        if index < self.validity_index - self.batch_size // self.mini_batch:
          
            self.validity_index = 0

        if index == self.validity_index:

            self.read_batch(index * self.batch_size, (index + 1) * self.batch_size)
            self.validity_index = (self.batch_size // self.mini_batch) + index

        # slices new batch
        store_index = (self.batch_size//self.mini_batch) - (self.validity_index - index)
       

        batch_images = self.image_batch_store[store_index, ...]
        batch_masks = self.mask_batch_store[store_index, ...]

        return tf.convert_to_tensor(batch_images), tf.convert_to_tensor(batch_masks)

    def on_epoch_end(self):
        # Shuffle image and mask filenames
      
        if self.shuffle:
   
            np.random.shuffle(self.image_filenames)
         