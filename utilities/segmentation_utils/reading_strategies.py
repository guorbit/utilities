from typing import Protocol
import os
import numpy as np
from PIL import Image
import rasterio

from flowreader import FlowGeneratorExperimental
from utilities.segmentation_utils import ImagePreprocessor


class ReaderInterface(Protocol):

    def read_batch(self, start:int, end: int) -> None:
        ...
    
    def get_dataset_size(self) -> None:
        ...

class RGB_Image_Strategy:

    def __init__(
        self,
        image_path: str,
        image_size: tuple [int, int],
        batch_image_filenames: np.ndarray,
    ):
        self.image_path = image_path
        self.image_size = image_size
        self.batch_image_files = batch_image_filenames

    def read_batch(self, batch_size, dataset_index) -> np.ndarray:
        #read images with PIL
        for i in range(batch_size):
            image_index = i + dataset_index
            image = Image.open(
                os.path.join(self.image_path, self.batch_image_filenames[image_index])
            ).resize(self.image_size, Image.ANTIALIAS)
            image = np.array(image)
            image = image / 255
        return image

    def get_dataset_size(self) -> int:
        dataset_size = FlowGeneratorExperimental.__len__
        return dataset_size

class Mask_Image_Strategy:

    def __init__(
        self,
        mask_path: str,
        batch_mask_filenames: np.ndarray,
        output_reshape: tuple[int, int],
    ):
        self.mask_path = mask_path
        self.batch_mask_filenames = batch_mask_filenames
        self.output_reshape = output_reshape
    
    def read_batch(self, batch_size, dataset_index) -> np.ndarray:
        #read images with PIL
        for i in range(batch_size):
            image_index = i + dataset_index
            mask = Image.open(
                os.path.join(self.mask_path, self.batch_mask_filenames[image_index])
                ).resize(self.output_reshape)
            mask = np.array(mask)
        return mask

    def get_dataset_size(self) -> int:
        dataset_size = FlowGeneratorExperimental.__len__
        return dataset_size
