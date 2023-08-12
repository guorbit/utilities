from typing import Protocol, Tuple
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
        mini_batch: int,
    ):
        self.image_path = image_path
        self.image_size = image_size
        self.mini_batch = mini_batch
        self.batch_image_files = batch_image_filenames

    def read_batch(self, batch_size, dataset_index) -> np.ndarray:
        #read images with PIL
        for i in range(batch_size):
                image_index = i + dataset_index
                image = Image.open(
                    os.path.join(self.image_path, self.batch_image_filenames[image_index])
                    ).resize(image_size, Image.ANTIALIAS)
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
    
    def read_batch(self, batch_size, dataset_index) -> np.ndarray:
        #read images with PIL
        for i in range(batch_size):
            for j in range(mini_batch):
                image_index = i * mini_batch + j
                mask = Image.open(
                    os.path.join(mask_path, batch_mask_filenames[image_index])
                    ).resize(output_reshape)
                mask = np.array(mask)
        return mask

    def get_dataset_size(self) -> int:
        dataset_size = FlowGeneratorExperimental.__len__
        return dataset_size



#should this be a batch with read_batch as the function having all the code in it?
    # def initialise_batch_img(self, mini_batch, image_size, channel_mask) -> np.ndarray:

    #     num_mini_batches = Reader.calculate_mini_batch
    #     channel_mask = np.array(channel_mask)
    #     n_channels = np.sum(channel_mask)

    #     batch_images = np.zeros(
    #         (
    #             num_mini_batches,
    #             mini_batch,
    #             image_size[0],
    #             image_size[1],
    #             n_channels,
    #         )
    #     )
    #     return batch_images

    # #output
    # def initialise_batch_mask(self, output_size, mini_batch, num_classes) -> Tuple[bool, np.ndarray]:
    #     #num_mini_batches = Reader.calculate_mini_batch

    #     if self.output_size[1] == 1:
    #         column = True
    #         batch_masks = np.zeros(
    #             (
    #                 num_mini_batches,
    #                 mini_batch, output_size[0],
    #                 num_classes
    #             )
    #         )
    #     else:
    #         column = False
    #         batch_masks = np.zeros(
    #             (
    #                 num_mini_batches,
    #                 mini_batch,
    #                 output_size[0],
    #                 output_size[1],
    #                 num_classes,
    #             )
    #         )

    #     return column, batch_masks
