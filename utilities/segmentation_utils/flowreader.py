"""
FlowGenerator is a wrapper around the keras ImageDataGenerator class.
"""

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from utilities.segmentation_utils import ImagePreprocessor


class FlowGenerator:
    """
    FlowGenerator is a wrapper around the keras ImageDataGenerator class.
    It can be used to read in images for semantic segmentation.
    Additionally, the reader can apply augmentation on the images,
    and one-hot encode them on the fly.
    """

    def __init__(
        self,
        image_path,
        mask_path,
        image_size,
        output_size,
        num_classes,
        shuffle=True,
        batch_size=2,
        seed=909,
    ):
        """
        Initializes the flow generator object,
        which can be used to read in images for semantic segmentation.
        Additionally, the reader can apply augmentation on the images,
        and one-hot encode them on the fly.

        Parameters:
        ----------
        image (string): path to the image directory
        mask (string): path to the mask directory
        batch_size (int): batch size
        image_size (tuple): image size
        output_size (tuple): output size
        #! Note: in case the output is a column vector it has to be in the shape (x, 1)
        num_classes (int): number of classes

        Keyword Arguments:
        -----------------
        shuffle (bool): whether to shuffle the dataset or not
        batch_size (int): batch size
        seed (int): seed for flow from directory

        Raises:
        ------
        ValueError: if the output size is not a tuple of length 2
        ValueError: if the output size is not a square matrix or a column vector

        Returns:
        -------
        None
        """

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
        self.__make_generator()
        print("Reading images from: ", self.image_path)

    def get_dataset_size(self):
        """
        Returns the length of the dataset

        Parameters:
        ----------
        None

        Returns:
        -------
        int: length of the dataset

        """

        return len(os.listdir(os.path.join(self.image_path, "img")))

    def __make_generator(self):
        """
        Creates the generator

        Parameters:
        ----------
        None

        Returns:
        -------
        None

        """

        image_datagen = ImageDataGenerator()
        mask_datagen = ImageDataGenerator()

        image_generator = image_datagen.flow_from_directory(
            self.image_path,
            class_mode=None,
            seed=self.seed,
            batch_size=self.batch_size,
            target_size=self.image_size,
        )

        mask_generator = mask_datagen.flow_from_directory(
            self.mask_path,
            class_mode=None,
            seed=self.seed,
            batch_size=self.batch_size,
            target_size=self.output_size,
            color_mode="grayscale",
        )

        self.train_generator = zip(image_generator, mask_generator)
        self.train_generator = self.preprocess(self.train_generator)

    def get_generator(self):
        """
        Returns the generator object

        Parameters:
        ----------
        None

        Returns:
        -------
        generator: generator object

        """
        return self.train_generator

    def preprocess(self, generator_zip, state=None):
        """
        Preprocessor function encapsulates both the image, and mask generator objects.
        Augments the images and masks and onehot encodes the masks

        Parameters:
        ----------
        generator_zip (tuple): tuple of image and mask generator

        Returns:
        -------
        a batch (tuple): generator batch of image and mask
        """
        for (img, mask) in generator_zip:
            for i in range(len(img)):
                # random state for reproducibility
                if state is None:
                    image_seed = np.random.randint(0, 100000)
                else:
                    state = np.random.RandomState(state)
                    image_seed = state.randint(0, 100000)

                img[i], mask[i] = ImagePreprocessor.augmentation_pipeline(
                    img[i], mask[i], self.image_size, self.output_size, seed=image_seed
                )
            mask = ImagePreprocessor.onehot_encode(
                mask, self.output_size, self.num_classes
            )
            yield (img, mask)
