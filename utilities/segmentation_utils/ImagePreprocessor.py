from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import numpy as np
import tensorflow as tf


class IPreprocessor(Protocol):
    """
    Interface of the preprocessing queue class
    Parameters
    ----------
    :queue list: list of functions to be applied
    """

    queue: list[Callable]

    def update_seed(self, seed: int) -> None:
        ...

    def get_queue_length(self) -> int:
        ...


class PreFunction:
    """
    Class that wraps a function and its arguments to be used in a preprocessing queue
    enables function to be defined with their parameters prior to being called.

    To call the function, simply call the PreFunction object with a tf.Tensor as an argument

    Parameters
    ----------
    :function Callable: function to be wrapped
    :args list: list of arguments to be passed to the function
    :kwargs dict: dictionary of keyword arguments to be passed to the function
    """

    def __init__(self, function: Callable, *args, **kwargs) -> None:
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def __call__(self, image: tf.Tensor) -> tf.Tensor:
        return self.function(image, *self.args, **self.kwargs)

    def set_seed(self, seed: int) -> None:
        """
        Changes the seed of the function

        Parameters
        ----------
        :seed int: seed to be changed to
        """
        self.kwargs["seed"] = seed


@dataclass
class PreprocessingQueue:
    """
    object to initialize a preprocessing queue

    Parameters
    ----------
    :queue list: list of functions to be applied

    """

    queue: list[PreFunction]

    def update_seed(self, seed):
        """
        Changes the seed of the queue

        Parameters
        ----------
        :seed int: seed to be changed to
        """
        for i in self.queue:
            i.set_seed(seed)

    def get_queue_length(self) -> int:
        """
        Returns the length of the queue

        Returns
        -------
        :return int: length of the queue
        """
        return len(self.queue)


def generate_image_queue(seed=0) -> PreprocessingQueue:
    """
    Generates the default image processing queue

    Keyword Arguments
    -----------------
    :seed int: seed to be used for the random functions

    Returns
    -------
    :return PreprocessingQueue: default queue
    """

    image_queue = PreprocessingQueue(
        [
            PreFunction(random_flip_left_right, seed=seed),
            PreFunction(random_flip_up_down, seed=seed),
            PreFunction(tf.image.random_brightness, max_delta=0.2, seed=seed),
            PreFunction(tf.image.random_contrast, lower=0.8, upper=1.2, seed=seed),
            PreFunction(tf.image.random_saturation, lower=0.8, upper=1.2, seed=seed),
        ],
    )
    return image_queue


def generate_mask_queue(seed=0) -> PreprocessingQueue:
    """
    Generates the default mask processing queue

    Keyword Arguments
    -----------------
    :seed int: seed to be used for the random functions

    Returns
    -------
    :return PreprocessingQueue: default queue
    """

    mask_queue = PreprocessingQueue(
        [
            PreFunction(random_flip_left_right, seed=seed),
            PreFunction(random_flip_up_down, seed=seed),
        ],
    )
    return mask_queue


def generate_default_queue(seed=0) -> tuple[PreprocessingQueue, PreprocessingQueue]:
    """
    Generates the default image and mask processing queues

    Keyword Arguments
    -----------------
    :seed int: seed to be used for the random functions

    Returns
    -------
    :return tuple(PreprocessingQueue, PreprocessingQueue): default queues
    """
    image_queue = generate_image_queue(seed)
    mask_queue = generate_mask_queue(seed)
    return image_queue, mask_queue


def onehot_encode(masks, num_classes) -> tf.Tensor:
    """
    Function that one-hot encodes masks

    :batch(tf.Tensor) masks: Masks to be encoded
    :tuple(int, int) output_size: Output size of the masks
    :int num_classes: Number of classes in the masks

    Returns
    -------
    :return tf.Tensor: Batch of one-hot encoded masks
    """
    #!TODO: add support for 1D masks
    encoded = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2], num_classes))
    for i in range(num_classes):
        mask = (masks == i).astype(float)
        encoded[:, :, :, i] = mask
    encoded = tf.convert_to_tensor(encoded)
    return encoded


def augmentation_pipeline(
    image,
    mask,
    image_queue: PreprocessingQueue,
    mask_queue: PreprocessingQueue,
    seed: int = 0,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Function that can execute a set of predifined augmentation functions
    stored in a PreprocessingQueue object. It augments both the image and the mask
    with the same functions and arguments.

    Parameters
    ----------
    :tf.Tensor image: The image to be processed
    :tf.Tensor mask: The mask to be processed


    Keyword Arguments
    -----------------
    :PreprocessingQueue, optional mask_queue image_queue: \
    Augmentation processing queue for images, defaults to None

    :PreprocessingQueue, optional mask_queue: Augmentation processing queue \
    for masks, defaults to None

    :int, optional seed: The seed to be used in the pipeline, defaults to 0

    Returns
    -------
    :return tuple(tf.Tensor, tf.Tensor): tuple of the processed image and mask
    """
    mask = tf.expand_dims(mask, axis=-1)

    image_queue.update_seed(seed)
    mask_queue.update_seed(seed)

    for fun_im, fun_mask in zip(image_queue.queue, mask_queue.queue):
        image = fun_im(image)
        mask = fun_mask(mask)



    mask = tf.squeeze(mask, axis=-1) # removes the last dimension
    mask = tf.convert_to_tensor(mask)
    # image = tf.convert_to_tensor(tf.clip_by_value(image, 0, 1))

    return image, mask


def flatten(image, input_size, channels=1) -> tf.Tensor:
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
    :return tf.Tensor: flattened image
    """
    # the 1 is required to preserve the shape similar to the original
    return tf.convert_to_tensor(
        tf.reshape(image, (input_size[0] * input_size[1], channels))
    )


def random_flip_up_down(image, seed=0) -> tf.Tensor:
    """
    Function that randomly flips an image up or down

    Parameters
    ----------
    :tf.Tensor image: image to be flipped

    Returns
    -------
    :return tf.Tensor: flipped image
    """

    state = np.random.RandomState(seed)
    flip = state.choice([True, False])
    if flip:
        return tf.convert_to_tensor(tf.image.flip_up_down(image))
    else:
        return image


def random_flip_left_right(image, seed=0) -> tf.Tensor:
    """
    Function that randomly flips an image left or right

    Parameters
    ----------
    :tf.Tensor image: image to be flipped

    Returns
    -------
    :return tf.Tensor: flipped image
    """

    state = np.random.RandomState(seed)
    flip = state.choice([True, False])
    if flip:
        return tf.convert_to_tensor(tf.image.flip_left_right(image))
    else:
        return image
