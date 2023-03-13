import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import Callable


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
    arguments: list[dict]


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


def augmentation_pipeline(image, mask, input_size,image_queue:PreprocessingQueue,mask_queue:PreprocessingQueue, channels=3):
    """
    Applies augmentation pipeline to the image and mask
    Parameters:
    ----------
    image (tf tensor): image to be augmented
    mask (tf tensor): mask to be augmented
    input_size (tuple): size of the input image
    Returns:
    -------
    image (tf tensor): augmented image
    mask (tf tensor): augmented mask
    """


    return image, mask


def flatten(image, input_size, channels=1):
    #!not tested
    return tf.reshape(image, (input_size[0] * input_size[1], channels))
