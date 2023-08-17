import os

import numpy as np
from PIL import Image
from pytest import MonkeyPatch

from utilities.segmentation_utils.reading_strategies import RGBImageStrategy


def test_read_batch_image_path() -> None:
    # should check if path is being read in correctly
    patch = MonkeyPatch()

    patch.setattr(os, "listdir", lambda x: ["a", "b", "c"])

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

    image_strategy = RGBImageStrategy(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        image_resample=Image.Resampling.NEAREST,
    )

    result = image_strategy.read_batch(batch_size=2, dataset_index=0)
    assert isinstance(result, np.ndarray)


def test_get_dataset_size() -> None:
    # checking if the calculation is done correctly

    image_strategy = RGBImageStrategy(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        image_resample=Image.Resampling.NEAREST,
    )
    dataset = 100  # if there are 100 images in the specified path
    mini_batch = 32  # and we want 32 images in each batch
    expected_value = dataset / mini_batch  # number of sets of images we expect

    dataset_size = image_strategy.get_dataset_size(mini_batch)
    assert dataset_size == expected_value
