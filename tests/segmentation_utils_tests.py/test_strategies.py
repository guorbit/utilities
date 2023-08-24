import os
import numpy as np
from PIL import Image
import rasterio
from pytest import MonkeyPatch

from utilities.segmentation_utils.reading_strategies import RGBImageStrategy


def test_read_batch_image_path() -> None:
    #checking if the file is being opened and read correctly
    patch = MonkeyPatch()

    mock_filenames = ["a", "b", "c"]

    patch.setattr(os, "listdir", lambda x: mock_filenames)

    patch.setattr(Image, "open", lambda _: Image.fromarray(np.ones((224, 224, 3)).astype(np.uint8)))

    image_strategy = RGBImageStrategy(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        image_resample=Image.Resampling.NEAREST,
    )

    batch_size = 2
    dataset_index = 0
    image_strategy.read_batch(batch_size, dataset_index)
    patch.undo()
    patch.undo()


def test_read_batch_returns_nparray() -> None:
    # checking if the returned value is a numpy array

    patch = MonkeyPatch()

    patch.setattr(os, "listdir", lambda x: ["a", "b", "c"])

    patch.setattr(Image, "open", lambda _: Image.fromarray(np.ones((224, 224, 3)).astype(np.uint8)))
    
    image_strategy = RGBImageStrategy(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        image_resample=Image.Resampling.NEAREST,
    )

    batch_size=2
    dataset_index=0

    result = image_strategy.read_batch(batch_size, dataset_index)
    assert isinstance(result, np.ndarray)

    patch.undo()
    patch.undo()


def test_get_dataset_size() -> None:
    # checking if the calculation is done correctly
    patch = MonkeyPatch()

    mock_filenames = ["a", "b", "c"]

    patch.setattr(os, "listdir", lambda x: mock_filenames)

    patch.setattr(Image, "open", lambda _: Image.fromarray(np.ones((224, 224, 3)).astype(np.uint8)))

    image_strategy = RGBImageStrategy(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        image_resample=Image.Resampling.NEAREST,
    )
    dataset = len(mock_filenames)  # number of images in the specified path
    mini_batch = 2  # number of images we want in each batch
    expected_value = int (np.floor(dataset / float(mini_batch)))  # number of sets of images we expect

    dataset_size = image_strategy.get_dataset_size(mini_batch)
    assert dataset_size == expected_value
    patch.undo()
    patch.undo()

class MockRasterio():
    
    def __init__(self):
        self.shape = (224, 224) #dimensions for raster data 
        self.dtypes = ['int32'] #data type of raster data that would be returned by .open()
        #a list containing a string representing a data type
        #32 bit int data type

    def read(self, *args, **kwargs):
        return np.zeros(self.shape, self.dtypes[0])
    
    #these functions are invoked when a 'with' statement is executed
    def __enter__(self):
        #called at the beginning of a 'with' block
        return self #returns instance of MockRasterio class itself
    
    def __exit__(self, type, value, traceback):
        #called at the end of a 'with' block
        pass

def test_hyperspectral_open():
        patch = MonkeyPatch()
        mock_filenames = ["a", "b", "c"]
        patch.setattr(os, "listdir", lambda x: mock_filenames)

        def mock_open(*args, **kwargs): #local function to the test
            #defines behaviour of mock object that replaces rasterio.open()
            return MockRasterio()
        
        patch.setattr(rasterio, "open", mock_open)
        image_path = "tests/segmentation_utils_tests/test_strategies"
        dataset_list = []

        for filename in mock_filenames:
            file_path = os.path.join(image_path, filename)
            dataset = rasterio.open(file_path)
            dataset_list.append(dataset)
    
            assert dataset.shape == (224, 224)
            assert np.array_equal (dataset.read(), np.zeros((224, 224), dtype='int32'))
