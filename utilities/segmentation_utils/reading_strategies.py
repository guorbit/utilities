from typing import Protocol
import os
import numpy as np
from PIL import Image
import rasterio

from flowreader import FlowGeneratorExperimental
from utilities.segmentation_utils import ImagePreprocessor


class IReader(Protocol):

    def read_batch(self, start:int, end: int) -> None:
        ...
    
    def get_dataset_size(self) -> None:
        ...

class RGB_Image_Strategy:

    def __init__(
        self,
        image_path: str,
        image_size: tuple [int, int],
    ):
        self.image_path = image_path
        self.image_size = image_size

    def read_batch(self, batch_size, dataset_index) -> np.ndarray:
        #read images with PIL
        image_filenames = np.array(sorted(os.listdir(self.image_path)))
        for i in range(batch_size):
            image_index = i + dataset_index
            image = Image.open(
                os.path.join(self.image_path, image_filenames[image_index])
            ).resize(self.image_size, Image.ANTIALIAS)
            image = np.array(image)
            image = image / 255
        return image

    def get_dataset_size(self, mini_batch) -> int:
        image_filenames = np.array(sorted(os.listdir(self.image_path)))
        dataset_size = int(np.floor(len(image_filenames) / float(mini_batch)))
        return dataset_size

class Mask_Image_Strategy:

    def __init__(
        self,
        mask_path: str,
        output_reshape: tuple[int, int],
    ):
        self.mask_path = mask_path
        self.output_reshape = output_reshape
    
    def read_batch(self, batch_size, dataset_index) -> np.ndarray:
        #read images with PIL
        mask_filenames = np.array(sorted(os.listdir(self.mask_path)))
        for i in range(batch_size):
            image_index = i + dataset_index
            mask = Image.open(
                os.path.join(self.mask_path, mask_filenames[image_index])
                ).resize(self.output_reshape)
            mask = np.array(mask)
        return mask

    def get_dataset_size(self, mini_batch) -> int:
        image_filenames = np.array(sorted(os.listdir(self.image_path)))
        dataset_size = int(np.floor(len(image_filenames) / float(mini_batch)))
        return dataset_size
