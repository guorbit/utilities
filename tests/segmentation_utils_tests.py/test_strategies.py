import os

import numpy as np
import pytest
from PIL import Image
from pytest import MonkeyPatch

from utilities.segmentation_utils.reading_strategies import (
    HSImageStrategy, HSImageStrategyMultiThread, RasterImageStrategy,
    RasterImageStrategyMultiThread, RGBImageStrategy)


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


class CV2Mock:
    IMREAD_UNCHANGED = 1
    COLOR_BGR2RGB = 1

    def __init__(self, n, size, bands) -> None:
        self.n = n
        self.size = size
        self.bands = bands
        self.call_count = 0

    def imread(self, *args, **kwargs):
        self.call_count += 1
        return np.full(
            (self.size[0], self.size[1], self.bands), self.call_count, np.uint8
        )

    def resize(self, *args, **kwargs):
        img = args[0]
        size = args[1]
        return np.full((size[0], size[1], self.bands), img[0, 0, 0], np.uint8)

    def cvtColor(self, *args, **kwargs):
        img = args[0]
        return np.full((self.size[0], self.size[1], self.bands), img[0, 0, 0], np.uint8)

    def get_count(self):
        return self.call_count


####################################################################################################
#                                     Package Mocks                                                #
####################################################################################################


@pytest.fixture
def rasterio_mock() -> MockRasterio:
    """
    Creates a mock of the rasterio package
    """
    return MockRasterio(n=3, size=(224, 224), bands=3, dtypes=["uint8"])


@pytest.fixture
def cv2_mock() -> CV2Mock:
    """
    Creates a mock of the cv2 package
    """
    return CV2Mock(n=3, size=(224, 224), bands=3)


####################################################################################################
#                                        OS mocks                                                  #
####################################################################################################


@pytest.fixture
def directory_mock(monkeypatch):
    """
    Mocks the os.listdir function to return a list of filenames
    """
    mock_filenames = [str(i) for i in range(20)]
    monkeypatch.setattr(os, "listdir", lambda x: mock_filenames)
    return len(mock_filenames)


@pytest.fixture
def mock_image_open(monkeypatch):
    """
    Mocks the Image.open function to return a numpy array
    """
    monkeypatch.setattr(
        Image,
        "open",
        lambda _: Image.fromarray(np.ones((224, 224, 3)).astype(np.uint8)),
    )


####################################################################################################
#                                    Strategy fixtures                                             #
####################################################################################################


@pytest.fixture
def rgb_strategy(mock_image_open) -> RGBImageStrategy:
    """
    Creates a RGBImageStrategy instance

    Relies on the mock_image_open fixture to mock the Image.open function
    """
    return RGBImageStrategy(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        image_resample=Image.Resampling.NEAREST,
    )


@pytest.fixture
def raster_strategy(rasterio_mock) -> RasterImageStrategy:
    """
    Creates a RasterImageStrategy instance

    Relies on the rasterio_mock fixture to mock the rasterio package
    """
    return RasterImageStrategy(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        package=rasterio_mock,
    )


@pytest.fixture
def raster_mt_strategy(rasterio_mock) -> RasterImageStrategyMultiThread:
    """
    Creates a RasterImageStrategyMultiThread instance

    Relies on the rasterio_mock fixture to mock the rasterio package
    """
    return RasterImageStrategyMultiThread(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        package=rasterio_mock,
    )


@pytest.fixture
def hsi_strategy(cv2_mock) -> HSImageStrategy:
    """
    Creates a HSImageStrategy instance

    Relies on the cv2_mock fixture to mock the cv2 package
    """
    return HSImageStrategy(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        package=cv2_mock,
    )


@pytest.fixture
def hsi_mt_strategy(cv2_mock) -> HSImageStrategyMultiThread:
    """
    Creates a HSImageStrategyMultiThread instance

    Relies on the cv2_mock fixture to mock the cv2 package
    """
    return HSImageStrategyMultiThread(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        package=cv2_mock,
    )


####################################################################################################
#                                 Test Generators                                                  #
####################################################################################################

FIXTURE_LIST = [
    "rgb_strategy",
    "raster_strategy",
    "raster_mt_strategy",
    "hsi_strategy",
    "hsi_mt_strategy",
]

FIXTURE_LIST_MT = [
    "raster_mt_strategy",
    "hsi_mt_strategy",
]


@pytest.fixture(params=FIXTURE_LIST)
def image_strategy(request, directory_mock):
    """
    Generates a strategy instance for each strategy type
    """
    strategy = request.getfixturevalue(request.param)
    return strategy


@pytest.fixture(params=FIXTURE_LIST_MT)
def mt_image_strategy(request, directory_mock):
    """
    Generates a strategy instance for each multi threaded strategy type
    """
    strategy = request.getfixturevalue(request.param)
    return strategy


@pytest.fixture(params=FIXTURE_LIST)
def fixture_factory(request, directory_mock):
    """
    Generates a strategy instance for each strategy type
    
    Can be used to generate multiple instances of the same strategy type
    """
    def make_instance():
        return request.getfixturevalue(request.param)

    return make_instance


@pytest.fixture(params=FIXTURE_LIST_MT)
def mt_fixture_factory(request, directory_mock):
    """
    Generates a strategy instance for each multi threaded strategy type

    Can be used to generate multiple instances of the same strategy type
    """
    def make_instance():
        return request.getfixturevalue(request.param)

    return make_instance


####################################################################################################
#                                 Test Functions                                                   #
####################################################################################################

@pytest.mark.development
def test_read_batch_image_path(image_strategy, mock_image_open) -> None:
    # checking if the file is being opened and read correctly

    strategy = image_strategy

    batch_size = 2
    dataset_index = 0
    result = strategy.read_batch(batch_size, dataset_index)

    assert result.shape == (2, 224, 224, 3)


@pytest.mark.development
def test_read_batch_returns_nparray(image_strategy) -> None:
    # checking if the returned value is a numpy array
    strategy = image_strategy

    batch_size = 2
    dataset_index = 0

    result = strategy.read_batch(batch_size, dataset_index)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 224, 224, 3)


@pytest.mark.development
def test_get_dataset_size(image_strategy, directory_mock) -> None:
    # checking if the calculation is done correctly

    strategy = image_strategy
    dataset = directory_mock  # number of images in the specified path
    mini_batch = 2  # number of images we want in each batch
    expected_value = int(
        np.floor(dataset / float(mini_batch))
    )  # number of sets of images we expect

    dataset_size = strategy.get_dataset_size(mini_batch)
    assert dataset_size == expected_value


@pytest.mark.development
def test_open(image_strategy):
    strategy = image_strategy

    read_images = strategy.read_batch(2, 0)

    assert read_images.shape == (2, 224, 224, 3)


@pytest.mark.development
def test_hsi_get_channels(directory_mock, hsi_strategy):
    strategy = hsi_strategy

    channels = strategy._HSImageStrategy__get_channels()

    assert channels == 3


@pytest.mark.development
def test_hsi_mt_get_channels(directory_mock, hsi_mt_strategy):
    strategy = hsi_mt_strategy

    channels = strategy._HSImageStrategyMultiThread__get_channels()

    assert channels == 3


@pytest.mark.development
def test_empty_batch(image_strategy):
    strategy = image_strategy

    batch_size = 0
    dataset_index = 0
    result = strategy.read_batch(batch_size, dataset_index)

    assert result.shape == (
        0,
        224,
        224,
        3,
    )  # 0 indicates there are no images in the batch


@pytest.mark.development
def test_out_of_bounds_index(image_strategy):
    strategy = RGBImageStrategy(
        image_path="tests/segmentation_utils_tests/test_strategies",
        image_size=(224, 224),
        image_resample=Image.Resampling.NEAREST,
    )

    batch_size = 2  # not an empty batch
    dataset_index = len(strategy.image_filenames)  # out of bounds index

    with pytest.raises(IndexError):
        strategy.read_batch(batch_size, dataset_index)


@pytest.mark.development
def test_batch_slicing(image_strategy):
    strategy = image_strategy

    batch_size = 10
    dataset_index = 2
    result = strategy.read_batch(batch_size, dataset_index)
    assert (
        result.shape[0] == batch_size
    )  # compare the size of returned data with batch_size


@pytest.mark.development
def test_get_image_size(image_strategy):
    strategy = image_strategy

    result = strategy.get_image_size()
    assert result == (224, 224)


@pytest.mark.development
def test_shuffle(fixture_factory):
    strategy_1 = fixture_factory()

    strategy_2 = fixture_factory()

    n = 100

    for i in range(n):
        strategy_1.shuffle_filenames(i)
        strategy_2.shuffle_filenames(i)

    assert np.array_equal(strategy_1.image_filenames, strategy_2.image_filenames)
    assert type(strategy_1) == type(strategy_2)


@pytest.mark.development
def test_mt_image_in_order(mt_image_strategy):
    strategy = mt_image_strategy

    batch_size = 10

    call_count = strategy.package.get_count()

    result = strategy.read_batch(batch_size, 0)

    for i in range(call_count, call_count + batch_size):
        assert np.array_equal(
            result[i - call_count, :, :, :], np.full((224, 224, 3), i + 1)
        )
