import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import Callable,Dict


@dataclass
class PreprocessingQueue:
    """
    object to initialize a preprocessing queue

    Parameters:
    ----------
    queue (list): list of functions to be applied

    Returns:
    -------
    None
    """
    queue: list[Callable]
    arguments: list[Dict]

    def update_seed(self, seed):
        """
        Changes the seed of the queue

        Parameters:
        ----------
        seed (int): seed to be changed to

        Returns:
        -------
        None
        """
        for i in self.arguments:
            i["seed"] = seed

    def get_queue_length(self):
        """
        Returns the length of the queue

        Parameters:
        ----------
        None

        Returns:
        -------
        int: length of the queue
        """
        return len(self.queue)
    
    


def generate_default_queue(seed = 0):
    """
    Generates the default processing queue

    Parameters:
    ----------
    None

    Returns:
    -------
    PreprocessingQueue: default queue
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
    return image_queue,mask_queue


def onehot_encode(masks, image_size, num_classes):
    """
    Onehot encodes the images coming from the image generator object
    Parameters:
    ----------
    masks (tf tensor): masks to be onehot encoded
    Returns:
    -------
    encoded (tf tensor): onehot encoded masks
    """
    encoded = np.zeros(
        (masks.shape[0], image_size[0] // 2 * image_size[1] // 2, num_classes)
    )
    for i in range(num_classes):
        encoded[:, :, i] = tf.squeeze((masks == i).astype(int))
    return encoded


def augmentation_pipeline(image, mask, input_size,image_queue:PreprocessingQueue = None,mask_queue:PreprocessingQueue = None, channels=3,seed = 0):
    """
    Applies augmentation pipeline to the image and mask
    If no queue is passed a default processing queue is created

    Parameters:
    ----------
    image (tf tensor): image to be augmented
    mask (tf tensor): mask to be augmented
    input_size (tuple): size of the input image

    Keyword Arguments:
    -----------------
    image_queue (PreprocessingQueue): queue of image processing functions
    mask_queue (PreprocessingQueue): queue of mask processing functions
    channels (int): number of channels in the image

    Raises:
    ------
    ValueError: if only one queue is passed

    Returns:
    -------
    image (tf tensor): augmented image
    mask (tf tensor): augmented mask
    """
    image_queue.update_seed(seed)
    mask_queue.update_seed(seed)
    
    if image_queue == None and mask_queue == None:
        image_queue, mask_queue = generate_default_queue()
    elif image_queue == None or mask_queue == None:
        raise ValueError("Both queues must be passed or none")

    for i,fun in enumerate(image_queue.queue):
        image = fun(image, **image_queue.arguments[i])

    for i,fun in enumerate(mask_queue.queue):
        mask = fun(mask, **mask_queue.arguments[i])

    return image, mask


def flatten(image, input_size, channels=1):
    #!not tested
    return tf.reshape(image, (input_size[0] * input_size[1], channels))
