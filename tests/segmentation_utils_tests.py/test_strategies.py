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


#!to be continued...
class MockRasterio():
    # def __init__(self, image_path, image_filenames):
    #     self.image_path = image_path
    #     self.image_filenames = image_filenames
    
    def __init__(self, func):
        self.func = func

    def mock_open(self, *args, **kwargs):
        patch = MonkeyPatch()
        mock_filenames = ["a", "b", "c"]
        patch.setattr(os, "listdir", lambda x: mock_filenames)

        image_file = os.path.join(self.image_path, self.image_filenames[image_index])
        dataset = rasterio.open(image_file)
        self.func(dataset)

    def mock_join(self):
        patch = MonkeyPatch()
        join = lambda x: "image_path"
        patch.setattr(os.path, "join", join)
        return join




def process_data(package=MockRasterio):
    package.open