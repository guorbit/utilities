import os

import numpy as np
import pytest
from PIL import Image
from pytest import MonkeyPatch

from utilities.segmentation_utils.reading_strategies import (
    HyperspectralImageStrategy, HyperspectralImageStrategyMultiThread,
    RGBImageStrategy)


class MockRasterio:
    def __init__(self, n, size, bands, dtypes):
        self.n = n
        self.size = size
        self.bands = bands
        self.dtypes = dtypes
        self.call_count = 0

    def open(self, *args, **kwargs):
        return self

    @property
    def count(self) -> int:
        return self.bands

    def read(self, *args, **kwargs):
        self.call_count += 1
        return np.full(
            (self.bands, self.size[0], self.size[1]), self.call_count, self.dtypes[0]
        )

    # these functions are invoked when a 'with' statement is executed
    def __enter__(self):
        # called at the beginning of a 'with' block
        return self  # returns instance of MockRasterio class itself

    def __exit__(self, type, value, traceback):
        # called at the end of a 'with' block
        pass

    def get_count(self):
        return self.call_count


@pytest.mark.development
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
    result = image_strategy.read_batch(batch_size, dataset_index)

    assert result.shape == (2, 224, 224, 3)
    patch.undo()
    patch.undo()


@pytest.mark.development
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
    assert result.shape == (2, 224, 224, 3)

    patch.undo()
    patch.undo()


@pytest.mark.development
def test_RGB_get_dataset_size() -> None:
    # checking if the calculation is done correctly
    patch = MonkeyPatch()

    mock_filenames = ["a", "b", "c"]

    patch.setattr(os, "listdir", lambda x: mock_filenames)

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


@pytest.mark.development
def test_Hyperspectral_get_dataset_size() -> None:
    # checking if the calculation is done correctly
    patch = MonkeyPatch()

    mock_filenames = ["a", "b", "c"]

    patch.setattr(os, "listdir", lambda x: mock_filenames)

    image_strategy = HyperspectralImageStrategy(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        package=MockRasterio(n=3, size=(224, 224), bands=3, dtypes=["uint8"]),
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


@pytest.mark.development
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


@pytest.mark.development
def test_hyperspectral_mt_open():
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
    strategy = HyperspectralImageStrategyMultiThread(
        image_path, (224, 224), package=MockRasterio(**mock_data)
    )

    read_images = strategy.read_batch(2, 0)

    assert read_images.shape == (2, 224, 224, 3)


@pytest.mark.development
def test_empty_batch():
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

    batch_size = 0
    dataset_index = 0
    result = image_strategy.read_batch(batch_size, dataset_index)

    assert result.shape == (
        0,
        224,
        224,
        3,
    )  # 0 indicates there are no images in the batch
    patch.undo()
    patch.undo()


@pytest.mark.development
def test_out_of_bounds_index():
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

    batch_size = 2  # not an empty batch
    dataset_index = len(image_strategy.image_filenames)  # out of bounds index

    try:
        image_strategy.read_batch(batch_size, dataset_index)
        assert True

    except IndexError:
        pass
    patch.undo()
    patch.undo()


@pytest.mark.development
def test_batch_slicing():
    patch = MonkeyPatch()

    mock_filenames = ["a" for _ in range(20)]

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

    batch_size = 10
    dataset_index = 2
    result = image_strategy.read_batch(batch_size, dataset_index)
    assert (
        result.shape[0] == batch_size
    )  # compare the size of returned data with batch_size
    patch.undo()
    patch.undo()


@pytest.mark.development
def test_RGB_get_image_size():
    patch = MonkeyPatch()

    mock_filenames = ["a" for _ in range(20)]

    patch.setattr(os, "listdir", lambda x: mock_filenames)

    image_strategy = RGBImageStrategy(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        image_resample=Image.Resampling.NEAREST,
    )

    result = image_strategy.get_image_size()
    assert result == (224, 224)


@pytest.mark.development
def test_HyperSpectral_get_image_size():
    patch = MonkeyPatch()

    mock_filenames = ["a" for _ in range(20)]

    patch.setattr(os, "listdir", lambda x: mock_filenames)

    image_strategy = HyperspectralImageStrategy(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        package=MockRasterio(n=3, size=(224, 224), bands=3, dtypes=["uint8"]),
    )

    result = image_strategy.get_image_size()
    assert result == (224, 224)


@pytest.mark.development
def test_HyperSpectral_MT_get_image_size():
    patch = MonkeyPatch()

    mock_filenames = ["a" for _ in range(20)]

    patch.setattr(os, "listdir", lambda x: mock_filenames)

    image_strategy = HyperspectralImageStrategyMultiThread(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        package=MockRasterio(n=3, size=(224, 224), bands=3, dtypes=["uint8"]),
    )

    result = image_strategy.get_image_size()
    assert result == (224, 224)

@pytest.mark.development
def test_RGB_shuffle():
    patch = MonkeyPatch()

    mock_filenames = [str(i) for i in range(20)]

    patch.setattr(os, "listdir", lambda x: mock_filenames)

    image_strategy_1 = RGBImageStrategy(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        image_resample=Image.Resampling.NEAREST,
    )

    image_strategy_2 = RGBImageStrategy(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        image_resample=Image.Resampling.NEAREST,
    )

    n = 100

    for i in range(n):
        image_strategy_1.shuffle_filenames(i)
        image_strategy_2.shuffle_filenames(i)

    assert np.array_equal(
        image_strategy_1.image_filenames, image_strategy_2.image_filenames
    )

@pytest.mark.development
def test_Hyperspectral_shuffle():
    patch = MonkeyPatch()

    mock_filenames = [str(i) for i in range(20)]

    patch.setattr(os, "listdir", lambda x: mock_filenames)

    image_strategy_1 = HyperspectralImageStrategy(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        package=MockRasterio(n=3, size=(224, 224), bands=3, dtypes=["uint8"]),
    )

    image_strategy_2 = HyperspectralImageStrategy(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        package=MockRasterio(n=3, size=(224, 224), bands=3, dtypes=["uint8"]),
    )

    n = 100

    for i in range(n):
        image_strategy_1.shuffle_filenames(i)
        image_strategy_2.shuffle_filenames(i)

    assert np.array_equal(
        image_strategy_1.image_filenames, image_strategy_2.image_filenames
    )

@pytest.mark.development
def test_Hyperspectral_mt_shuffle():
    patch = MonkeyPatch()

    mock_filenames = [str(i) for i in range(20)]

    patch.setattr(os, "listdir", lambda x: mock_filenames)

    image_strategy_1 = HyperspectralImageStrategyMultiThread(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        package=MockRasterio(n=3, size=(224, 224), bands=3, dtypes=["uint8"]),
    )

    image_strategy_2 = HyperspectralImageStrategy(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        package=MockRasterio(n=3, size=(224, 224), bands=3, dtypes=["uint8"]),
    )

    n = 100

    for i in range(n):
        image_strategy_1.shuffle_filenames(i)
        image_strategy_2.shuffle_filenames(i)

    assert np.array_equal(
        image_strategy_1.image_filenames, image_strategy_2.image_filenames
    )

@pytest.mark.development
def test_Hyperspectral_mt_image_in_order():
    patch = MonkeyPatch()

    mock_filenames = [str(i) for i in range(20)]

    patch.setattr(os, "listdir", lambda x: mock_filenames)
    mock_package = MockRasterio(n=3, size=(224, 224), bands=3, dtypes=["uint8"])
    image_strategy = HyperspectralImageStrategyMultiThread(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        package=mock_package,
    )

    batch_size = 10

    call_count = mock_package.get_count()

    result = image_strategy.read_batch(batch_size, 0)

    for i in range(call_count, call_count + batch_size):
        assert np.array_equal(
            result[i - call_count, :, :, :], np.full((224, 224, 3), i + 1)
        )
