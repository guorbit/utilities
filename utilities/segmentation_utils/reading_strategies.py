import os
from typing import Protocol

import numpy as np
import rasterio
from PIL import Image


class IReader(Protocol):

    def read_batch(self, start:int, end: int) -> None:
        ...
    
    def get_dataset_size(self) -> None:
        ...

class RGBImageStrategy:

    def __init__(
        self,
        image_path: str,
        image_size: tuple[int, int],
        image_resample = Image.Resampling.NEAREST,
    ):
        self.image_path = image_path
        self.image_size = image_size
        self.image_resample = image_resample

    def read_batch(self, batch_size, dataset_index) -> np.ndarray:
        #read images with PIL

        #! add this to the intializer
        image_filenames = np.array(sorted(os.listdir(self.image_path)))
        for i in range(batch_size):
            image_index = i + dataset_index
            image = Image.open(
                os.path.join(self.image_path, image_filenames[image_index])
            ).resize(self.image_size, self.image_resample)
            image = np.array(image)
            image = image / 255
        return image

    def get_dataset_size(self, mini_batch) -> int:
        image_filenames = np.array(sorted(os.listdir(self.image_path)))
        dataset_size = int(np.floor(len(image_filenames) / float(mini_batch)))
        return dataset_size
    
class HyperspectralImageStrategy:

    def __init__(
        self,
        image_path:str,
        image_resize:tuple[int,int],
        image_resample = Image.Resampling.NEAREST,
        
    ):
        self.image_path = image_path
        self.image_resize = image_resize
        self.image_resample = image_resample
  
    def read_batch(self, batch_size, dataset_index) -> np.ndarray:
        #read images with rasterio
        image_filenames = np.array(sorted(os.listdir(self.image_path)))
        for i in range(batch_size):
            image_index = i + dataset_index
            #open the source raster dataset
            with rasterio.open(
                os.path.join(self.image_path, image_filenames[image_index])
                ) as dataset:
                 #.read() returns a numpy array that contains the raster cell values in your file.
                image = dataset.read()
                image = image / 255
                image = np.resize(self.image_resize, self.image_resample)
        return image
    
    def get_dataset_size(self, mini_batch) -> int:
        image_filenames = np.array(sorted(os.listdir(self.image_path)))
        dataset_size = int(np.floor(len(image_filenames) / float(mini_batch)))
        return dataset_size
    