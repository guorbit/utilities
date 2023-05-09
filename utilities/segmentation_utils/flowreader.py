"""
FlowGenerator is a wrapper around the keras ImageDataGenerator class.
"""

import math
import os
from typing import Optional

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

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
    :PreprocessingQueue preprocessing_queue_image: preprocessing queue for images
    :PreprocessingQueue preprocessing_queue_mask: preprocessing queue for masks

    Raises
    ------
    :ValueError: if the output size is not a tuple of length 2
    :ValueError: if the output size is not a square matrix or a column vector
    """

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
        preprocessing_queue_image: Optional[ImagePreprocessor.PreprocessingQueue] = None,
        preprocessing_queue_mask: Optional[ImagePreprocessor.PreprocessingQueue] = None,
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
        self.preprocessing_queue_image = preprocessing_queue_image
        self.preprocessing_queue_mask = preprocessing_queue_mask
        self.preprocessing_seed = preprocessing_seed
        self.__make_generator()
        print("Reading images from: ", self.image_path)

    def get_dataset_size(self):
        """
        Returns the length of the dataset

        Returns
        -------
        :returns int: length of the dataset
        """

        return len(os.listdir(os.path.join(self.image_path, "img")))

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
            class_mode=None, # type: ignore
            seed=self.seed,
            batch_size=self.batch_size,
            target_size=self.image_size,
        )

        mask_generator = mask_datagen.flow_from_directory(
            self.mask_path,
            class_mode=None, # type: ignore
            seed=self.seed,
            batch_size=self.batch_size,
            target_size=self.output_size,
            color_mode="grayscale",
        )
        if self.preprocessing_queue_image is None and self.preprocessing_queue_mask is None:
            #!Possibly in the wrong place as it has to be regenerated every time
            (
                self.preprocessing_queue_image,
                self.preprocessing_queue_mask,
            ) = ImagePreprocessor.generate_default_queue()
        elif self.preprocessing_queue_image is None or self.preprocessing_queue_mask is None:
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
        for (img, mask) in generator_zip:
            if self.preprocessing_enabled:
                for i_image,i_mask in zip(img, mask):
                    # random state for reproducibility
                    if self.preprocessing_seed is None:
                        image_seed = np.random.randint(0, 100000)
                    else:
                        state = np.random.RandomState(self.preprocessing_seed)
                        image_seed = state.randint(0, 100000)

                    i_image, i_mask = ImagePreprocessor.augmentation_pipeline(
                        image = i_image,
                        mask = i_mask,
                        input_size = self.image_size,
                        output_size = self.output_size,
                        output_reshape = self.output_reshape,
                        seed=image_seed,
                        #!both preprocessing queues are assigned by this time
                        image_queue=self.preprocessing_queue_image, # type: ignore
                        mask_queue=self.preprocessing_queue_mask, # type: ignore
                    )
            mask = ImagePreprocessor.onehot_encode(
                mask, self.output_size, self.num_classes
            )
            yield (img, mask)
