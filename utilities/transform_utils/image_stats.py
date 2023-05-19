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
    :tupple(dict, dict): a tuple containing the distribution of classes and the number of files containing each class
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
        max = size[0] * size[1]
        img = np.array(img)
        for i in range(num_classes):
            sum = np.sum(img == i)
            distribution[i] += sum / max
            if sum > 0:
                df[i] += 1

    return distribution, df



if __name__ == "__main__":
    path = "cut_masks"
    files = os.listdir(path)
    dist = get_distribution_seg(path, files)
    print(dist)
