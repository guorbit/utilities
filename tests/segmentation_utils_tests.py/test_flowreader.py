import os

import numpy as np
import pytest
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from pytest import MonkeyPatch

from utilities.segmentation_utils import ImagePreprocessor
from utilities.segmentation_utils.flowreader import FlowGeneratorExperimental


def test_can_create_instance() -> None:
    patch = MonkeyPatch()
    # mock list directory
    patch.setattr(os, "listdir", lambda x: ["a", "b", "c"])

    # create generator instance
    generator = FlowGeneratorExperimental(
        image_path="tests/segmentation_utils_tests/flow_reader_test",
        mask_path="tests/segmentation_utils_tests/flow_reader_test",
        image_size=(512, 512),
        output_size=(512,512),
        num_classes=7,
        channel_mask= [True,True,True]
    )
    pass

def test_set_preprocessing_pipeline() -> None:
    patch = MonkeyPatch()
    # mock list directory
    patch.setattr(os, "listdir", lambda x: ["a", "b", "c"])

    # create generator instance
    generator = FlowGeneratorExperimental(
        image_path="tests/segmentation_utils_tests/flow_reader_test",
        mask_path="tests/segmentation_utils_tests/flow_reader_test",
        image_size=(512, 512),
        output_size=(512,512),
        num_classes=7,
        channel_mask= [True,True,True]
    )

    image_queue = ImagePreprocessor.PreprocessingQueue(queue=[],arguments=[])
    mask_queue = ImagePreprocessor.PreprocessingQueue(queue=[],arguments=[])

    generator.set_preprocessing_pipeline(
        image_queue,mask_queue
    )
    pass

def test_set_mini_batch_size() -> None:
    patch = MonkeyPatch()
    # mock list directory
    patch.setattr(os, "listdir", lambda x: ["a", "b", "c"])

    # create generator instance
    generator = FlowGeneratorExperimental(
        image_path="tests/segmentation_utils_tests/flow_reader_test",
        mask_path="tests/segmentation_utils_tests/flow_reader_test",
        image_size=(512, 512),
        output_size=(512,512),
        num_classes=7,
        channel_mask= [True,True,True]
    )

    generator.set_mini_batch_size(2)
    assert generator.mini_batch == 2

def test_set_mini_batch_size_too_large() -> None:

    patch = MonkeyPatch()
    # mock list directory
    patch.setattr(os, "listdir", lambda x: ["a", "b", "c"])

    # create generator instance
    generator = FlowGeneratorExperimental(
        image_path="tests/segmentation_utils_tests/flow_reader_test",
        mask_path="tests/segmentation_utils_tests/flow_reader_test",
        image_size=(512, 512),
        output_size=(512,512),
        num_classes=7,
        channel_mask= [True,True,True]
    )
    with pytest.raises(ValueError) as exc_info:
        generator.set_mini_batch_size(5)

    assert exc_info.value.args[0] == "The mini batch size cannot be larger than the batch size"


def test_set_mini_batch_size_not_devisable() -> None:

    patch = MonkeyPatch()
    # mock list directory
    patch.setattr(os, "listdir", lambda x: ["a", "b", "c"])

    # create generator instance
    generator = FlowGeneratorExperimental(
        image_path="tests/segmentation_utils_tests/flow_reader_test",
        mask_path="tests/segmentation_utils_tests/flow_reader_test",
        image_size=(512, 512),
        output_size=(512,512),
        num_classes=7,
        channel_mask= [True,True,True],
        batch_size=3
        
    )
    with pytest.raises(ValueError) as exc_info:
        generator.set_mini_batch_size(2)

    assert exc_info.value.args[0] == "The batch size must be divisible by the mini batch size"
    
