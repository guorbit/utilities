import os

import numpy as np
import rasterio
from PIL import Image
from pytest import MonkeyPatch

from utilities.segmentation_utils.reading_strategies import (
    HyperspectralImageStrategy, MockRasterio, RGBImageStrategy)


def test_read_batch_image_path() -> None:
    # checking if the file is being opened and read correctly
    patch = MonkeyPatch()

    mock_filenames = ["a", "b", "c"]

    patch.setattr(os, "listdir", lambda x: mock_filenames)

    patch.setattr(
        Image,
        "open",
        lambda _: Image.fromarray(np.ones((224, 224, 3)).astype(np.uint8)),
    )

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

    patch.setattr(
        Image,
        "open",
        lambda _: Image.fromarray(np.ones((224, 224, 3)).astype(np.uint8)),
    )

    image_strategy = RGBImageStrategy(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        image_resample=Image.Resampling.NEAREST,
    )

    batch_size = 2
    dataset_index = 0

    result = image_strategy.read_batch(batch_size, dataset_index)
    assert isinstance(result, np.ndarray)

    patch.undo()
    patch.undo()


def test_get_dataset_size() -> None:
    # checking if the calculation is done correctly
    patch = MonkeyPatch()

    mock_filenames = ["a", "b", "c"]

    patch.setattr(os, "listdir", lambda x: mock_filenames)

    #! not needed as you arent reading any image in this function
    patch.setattr(
        Image,
        "open",
        lambda _: Image.fromarray(np.ones((224, 224, 3)).astype(np.uint8)),
    )

    image_strategy = RGBImageStrategy(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        image_resample=Image.Resampling.NEAREST,
    )
    dataset = len(mock_filenames)  # number of images in the specified path
    mini_batch = 2  # number of images we want in each batch
    expected_value = int(
        np.floor(dataset / float(mini_batch))
    )  # number of sets of images we expect

    dataset_size = image_strategy.get_dataset_size(mini_batch)
    assert dataset_size == expected_value
    patch.undo()
    patch.undo()


def test_hyperspectral_open():
    patch = MonkeyPatch()
    mock_filenames = ["a", "b", "c"]
    patch.setattr(os, "listdir", lambda x: mock_filenames)

    image_path = "tests/segmentation_utils_tests/test_strategies"
    
    mock_data = {
        "n": 3,
        "size": (224, 224),
        "bands": 3,
        "dtypes": ["uint8"],
    }
    strategy = HyperspectralImageStrategy(
        image_path, (224, 224), package=MockRasterio(**mock_data)
    )

    read_images = strategy.read_batch(2, 0)

    assert read_images.shape == (2, 224, 224, 3)
