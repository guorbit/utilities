import os
from utilities.segmentation_utils.flowreader import FlowGenerator
from keras.preprocessing.image import ImageDataGenerator
from utilities.segmentation_utils import ImagePreprocessor
from pytest import MonkeyPatch
import numpy as np

# mock implementations
def flow_from_directory_mock(*args, **kwargs):
    channels = 3
    if "color_mode" in kwargs and kwargs["color_mode"] == "grayscale":
        channels = 1

    batch = np.zeros((2, kwargs["target_size"][0], kwargs["target_size"][1], channels))
    return batch


generator_args = {
    "image_path": "tests/segmentation_utils_tests/flow_reader_test",
    "mask_path": "tests/segmentation_utils_tests/flow_reader_test",
    "image_size": (512, 512),
    "output_size": (256 * 256, 1),
    "num_classes": 7,
    "shuffle": True,
    "batch_size": 2,
    "seed": 909,
}

mock_onehot_fn = lambda x, y, z: np.rollaxis(np.array([x for i in range(z)]), 0, 3)
mock_augmentation_fn = lambda x, y, z, a, b: (x, y)

# tests
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
    flow_generator = FlowGenerator(**generator_args)
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
    patch.setattr(ImagePreprocessor, "augmentation_pipeline", mock_augmentation_fn)
    patch.setattr(
        ImagePreprocessor,
        "onehot_encode",
        mock_onehot_fn,
    )

    # create a flow generator
    flow_generator = FlowGenerator(**generator_args)
    patch.undo()
    patch.undo()


def test_get_dataset_size() -> None:
    patch = MonkeyPatch()
    patch.setattr(os, "listdir", lambda x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # mock an imagedatagenerator from keras
    mock_image_datagen = patch.setattr(
        ImageDataGenerator,
        "flow_from_directory",
        flow_from_directory_mock,
    )
    # mock external dependencies
    patch.setattr(ImagePreprocessor, "augmentation_pipeline", mock_augmentation_fn)
    patch.setattr(
        ImagePreprocessor,
        "onehot_encode",
        mock_onehot_fn,
    )
    # create a flow generator
    flow_generator = FlowGenerator(**generator_args)
    size = flow_generator.get_dataset_size()
    assert size == 10
    patch.undo()
    patch.undo()
    patch.undo()


def test_get_generator() -> None:
    patch = MonkeyPatch()

    # mock external dependencies
    patch.setattr(ImagePreprocessor, "augmentation_pipeline", mock_augmentation_fn)
    patch.setattr(
        ImagePreprocessor,
        "onehot_encode",
        mock_onehot_fn,
    )

    # create a flow generator
    flow_generator = FlowGenerator(**generator_args)
    generator = flow_generator.get_generator()

    assert generator != None
    patch.undo()
    patch.undo()
