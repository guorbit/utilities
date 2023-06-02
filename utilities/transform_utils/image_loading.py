"""
This module contains functions for loading images from a directory into a numpy array.
"""


import os
import numpy as np
from numpy.typing import NDArray
from PIL import Image


def load_ims(ims_path: str) -> NDArray[np.int16]:
    """
    Loads images from the input directory into numpy tensor.
    Specfically dimensions are [N, H, W, C].
    where   N = Number of images
            H = Image height
            W = Images Width
            C = Number of channels/bands of image

    Parameters
    ----------
    :str ims_path: directory where images are stored.

    Returns
    -------
    :return NDArray[np.int16]: numpy array of images in format [N, H, W, C]
    """
    dir_contents = os.listdir(ims_path)
    num_ims = len(dir_contents)
    image = Image.open(ims_path + "\\" + dir_contents[0])
    width, height = image.size
    im_bands = len(image.getbands())

    ims = np.empty((num_ims, height, width, im_bands), np.int16)

    for index in range(num_ims):
        ims[index, :, :, :] = np.array(
            Image.open(ims_path + "\\" + dir_contents[index]), np.int16
        )

    return ims
