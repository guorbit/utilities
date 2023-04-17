import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def get_distribution_seg(path,files:list[str], num_classes:int=7):
    set_of_files = set(files)
    distribution = {}
    for i in range(num_classes):
        distribution[i] = 0

    for file in tqdm(set_of_files):
        img = Image.open(os.path.join(path,file))
        img = np.array(img)
        for i in range(num_classes):
            distribution[i] += np.sum(img==i)
    
    return distribution

if __name__ == "__main__":
    path = "cut_masks"
    files = os.listdir(path)
    dist = get_distribution_seg(path,files)
    print(dist)