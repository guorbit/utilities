import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def get_distribution_seg(path: str, files: list[str], num_classes: int = 7):
    """
    Returns the distribution of classes in the segmentation dataset,
    and the number of files the class is present in.

    Parameters
    ----------
    :param path: path to the directory containing the segmentation masks
    :param files: list of files in the directory

    Keyword Arguments
    -----------------
    :param num_classes: number of classes in the dataset

    Returns
    -------
    :return tupple(dict,dict): dictionary containing the distribution of classes
    """
    set_of_files = set(files)
    distribution = {}
    for i in range(num_classes):
        distribution[i] = 0

    for file in tqdm(set_of_files):
        img = Image.open(os.path.join(path, file))
        img = np.array(img)
        for i in range(num_classes):
            distribution[i] += np.sum(img == i)
    return distribution


if __name__ == "__main__":
    mask_path = "cut_masks"
    mask_files = os.listdir(mask_path)
    dist = get_distribution_seg(mask_path, mask_files)
    print(dist)
