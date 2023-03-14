import os
from utilities.segmentation_utils.flowreader import FlowGenerator
from keras.preprocessing.image import ImageDataGenerator
from utilities.segmentation_utils import ImagePreprocessor
from pytest import MonkeyPatch
import numpy as np


def flow_from_directory_mock(*args, **kwargs):
    channels = 3
    if "color_mode" in kwargs and kwargs["color_mode"] == "grayscale":
        channels = 1

    batch = np.zeros((2, kwargs["target_size"][0], kwargs["target_size"][1], channels))
    return batch


def test_makes_flow_generator() -> None:
    patch = MonkeyPatch()
    # mock an imagedatagenerator from keras
    mock_image_datagen = patch.setattr(
        ImageDataGenerator,
        "flow_from_directory",
        flow_from_directory_mock,
    )
    patch.setattr(FlowGenerator, "preprocess", lambda self, x: x)

    # create a flow generator
    flow_generator = FlowGenerator(
        image_path="tests/segmentation_utils_tests/flow_reader_test",
        mask_path="tests/segmentation_utils_tests/flow_reader_test",
        image_size=(512, 512),
        num_classes=7,
        shuffle=True,
        batch_size=2,
        seed=909,
    )
    pass


def test_flow_generator_with_preprocess() -> None:
    patch = MonkeyPatch()
    # mock an imagedatagenerator from keras
    mock_image_datagen = patch.setattr(
        ImageDataGenerator,
        "flow_from_directory",
        flow_from_directory_mock,
    )
    # mock external dependencies
    patch.setattr(ImagePreprocessor, "augmentation_pipeline", lambda x, y, z, a, b: (x, y))
    patch.setattr(ImagePreprocessor, "onehot_encode", lambda x, y, z: np.rollaxis(np.array([x for i in range(z)]),0,3))
    # create a flow generator
    flow_generator = FlowGenerator(
        image_path="tests/segmentation_utils_tests/flow_reader_test",
        mask_path="tests/segmentation_utils_tests/flow_reader_test",
        image_size=(512, 512),
        num_classes=7,
        shuffle=True,
        batch_size=2,
        seed=909,
    )
    patch.undo()
    patch.undo()

    

