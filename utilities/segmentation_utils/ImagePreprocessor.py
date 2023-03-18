from dataclasses import dataclass
from typing import Callable, Dict
import numpy as np
import tensorflow as tf



@dataclass
class PreprocessingQueue:
    """
    object to initialize a preprocessing queue

    Parameters
    ----------
    :queue list: list of functions to be applied
    :arguments list[dict]: list of arguments to be passed to the functions
    """

    queue: list[Callable]
    arguments: list[Dict]

    def update_seed(self, seed):
        """
        Changes the seed of the queue

        Parameters
        ----------
        :seed int: seed to be changed to
        """
        for i in self.arguments:
            i["seed"] = seed

    def get_queue_length(self):
        """
        Returns the length of the queue

        Returns
        -------
        :return: length of the queue
        :rtype: int
        """
        return len(self.queue)


def generate_default_queue(seed=0):
    """
    Generates the default processing queue

    Keyword Arguments
    -----------------
    :seed int: seed to be used for the random functions

    Returns
    -------
    :return PreprocessingQueue: default queue
    """
    image_queue = PreprocessingQueue(
        queue=[
            tf.image.random_flip_left_right,
            tf.image.random_flip_up_down,
            tf.image.random_brightness,
            tf.image.random_contrast,
            tf.image.random_saturation,
            tf.image.random_hue,
        ],
        arguments=[
            {"seed": seed},
            {"seed": seed},
            {"max_delta": 0.2, "seed": seed},
            {"lower": 0.8, "upper": 1.2, "seed": seed},
            {"lower": 0.8, "upper": 1.2, "seed": seed},
            {"max_delta": 0.2, "seed": seed},
        ],
    )
    mask_queue = PreprocessingQueue(
        queue=[
            tf.image.random_flip_left_right,
            tf.image.random_flip_up_down,
        ],
        arguments=[
            {"seed": seed},
            {"seed": seed},
        ],
    )
    return image_queue, mask_queue


def onehot_encode(masks, output_size, num_classes):
    """
    Function that one-hot encodes masks

    :batch(tf.Tensor) masks: Masks to be encoded
    :tuple(int, int) output_size: Output size of the masks
    :int num_classes: Number of classes in the masks

    Returns
    -------
    :return: Encoded masks
    :rtype: batch(tf.Tensor)
    """
    encoded = np.zeros((masks.shape[0], output_size[0] * output_size[1], num_classes))
    for i in range(num_classes):
        encoded[:, :, i] = tf.squeeze((masks == i).astype(int))
    return encoded


def augmentation_pipeline(
    image,
    mask,
    input_size: tuple[int, int],
    output_size: tuple[int, int],
    output_reshape: tuple[int, int] = None,
    image_queue: PreprocessingQueue = None,
    mask_queue: PreprocessingQueue = None,
    channels: int = 3,
    seed: int = 0,
):
    """
    Function that can execute a set of predifined augmentation functions
    stored in a PreprocessingQueue object. It augments both the image and the mask
    with the same functions and arguments.

    Parameters
    ----------
    :tf.Tensor image: The image to be processed
    :tf.Tensor mask: The mask to be processed
    :tuple(int, int) input_size: Input size of the image
    :tuple(int, int) output_size: Output size of the image


    Keyword Arguments
    -----------------
    :tuple(int, int), optional output_reshape: In case the image is a column vector,
    this is the shape it should be reshaped to. Defaults to None.
    
    :PreprocessingQueue, optional mask_queue image_queue: 
    Augmentation processing queue for images, defaults to None

    :PreprocessingQueue, optional mask_queue: Augmentation processing queue
    for masks, defaults to None
    
    :int, optional channels: Number of bands in the image, defaults to 3
    :int, optional seed: The seed to be used in the pipeline, defaults to 0

    Raises
    ------
    :raises ValueError: If only one of the queues is passed

    Returns
    -------
    :return: tuple of the processed image and mask
    :rtype: tuple(tf.Tensor, tf.Tensor)
    """

    # reshapes masks, such that transforamtions work properly
    if output_size[1] == 1:
        mask = tf.reshape(mask, (output_reshape[0], output_reshape[1], 1))

    image_queue.update_seed(seed)
    mask_queue.update_seed(seed)

    for i, fun in enumerate(image_queue.queue):
        image = fun(image, **image_queue.arguments[i])

    for i, fun in enumerate(mask_queue.queue):
        mask = fun(mask, **mask_queue.arguments[i])

    # flattens masks out to the correct output shape
    if output_size[1] == 1:
        mask = flatten(mask, output_size, channels=1)
    return image, mask


def flatten(image, input_size, channels=1):
    """flatten
    Function that flattens an image preserving the number of channels

    Parameters
    ----------
    :tf.Tensor image: image to be flattened
    :tuple(int, int) input_size: input size of the image

    Keyword Arguments
    -----------------
    :int, optional channels: number of chanels to preserve, defaults to 1

    Returns
    -------
    :return: flattened image
    :rtype: tf.Tensor
    """
    # the 1 is required to preserve the shape similar to the original
    return tf.reshape(image, (input_size[0] * input_size[1], 1, channels))
