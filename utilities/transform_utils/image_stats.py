import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def get_distribution_seg(path, files: list[str], num_classes: int = 7):
    """
    Returns the distribution of classes in the segmentation dataset

    Arguments
    ---------
    :str path: path to the folder containing the segmentation masks
    :list[str] files: list of files in the folder

    Keyword Arguments
    -----------------
    :int, optional num_classes: number of classes in the dataset, defaults to 7

    Returns
    -------
    :tupple(dict, dict): a tuple containing the distribution of \
    classes and the number of files containing each class
    """

    set_of_files = set(files)
    distribution = {}
    df = {}
    for i in range(num_classes):
        distribution[i] = 0
        df[i] = 0

    for file in tqdm(set_of_files):
        img = Image.open(os.path.join(path, file))
        size = img.size
        max_pixels = size[0] * size[1]
        img = np.array(img)
        for i in range(num_classes):
            class_sum = np.sum(img == i)
            distribution[i] += class_sum / max_pixels
            if class_sum > 0:
                df[i] += 1

    return distribution, df



if __name__ == "__main__":
    mask_path = "cut_masks"
    mask_files = os.listdir(mask_path)
    dist = get_distribution_seg(mask_path, mask_files)
    print(dist)
