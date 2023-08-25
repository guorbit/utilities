import os

import numpy as np
import pytest
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from pytest import MonkeyPatch

from utilities.segmentation_utils import ImagePreprocessor
from utilities.segmentation_utils.flowreader import FlowGeneratorExperimental


class DummyStrategy:
    def __init__(self, input_shape=(512, 512, 3)):
        self.input_shape = input_shape

    def read_batch(self, batch_size: int, dataset_index: int) -> np.ndarray:
        return np.zeros((batch_size, *self.input_shape))

    def get_dataset_size(self) -> int:
        return 10


@pytest.mark.development
def test_can_create_instance() -> None:
    patch = MonkeyPatch()
    # mock list directory
    patch.setattr(os, "listdir", lambda x: ["a", "b", "c"])

    input_strategy = DummyStrategy()
    output_strategy = DummyStrategy()

    # create generator instance
    generator = FlowGeneratorExperimental(
        image_path="tests/segmentation_utils_tests/flow_reader_test",
        mask_path="tests/segmentation_utils_tests/flow_reader_test",
        image_size=(512, 512),
        output_size=(512, 512),
        num_classes=7,
        channel_mask=[True, True, True],
        input_strategy=input_strategy,
        output_strategy=output_strategy,
    )
    pass


@pytest.mark.development
def test_set_preprocessing_pipeline() -> None:
    patch = MonkeyPatch()
    # mock list directory
    patch.setattr(os, "listdir", lambda x: ["a", "b", "c"])

    input_strategy = DummyStrategy()
    output_strategy = DummyStrategy()
    # create generator instance
    generator = FlowGeneratorExperimental(
        image_path="tests/segmentation_utils_tests/flow_reader_test",
        mask_path="tests/segmentation_utils_tests/flow_reader_test",
        image_size=(512, 512),
        output_size=(512, 512),
        num_classes=7,
        channel_mask=[True, True, True],
        input_strategy=input_strategy,
        output_strategy=output_strategy,
    )

    image_queue = ImagePreprocessor.PreprocessingQueue(queue=[])
    mask_queue = ImagePreprocessor.PreprocessingQueue(queue=[])

    generator.set_preprocessing_pipeline(image_queue, mask_queue)
    pass


@pytest.mark.development
def test_set_mini_batch_size() -> None:
    patch = MonkeyPatch()
    # mock list directory
    patch.setattr(os, "listdir", lambda x: ["a", "b", "c"])

    input_strategy = DummyStrategy()
    output_strategy = DummyStrategy()

    # create generator instance
    generator = FlowGeneratorExperimental(
        image_path="tests/segmentation_utils_tests/flow_reader_test",
        mask_path="tests/segmentation_utils_tests/flow_reader_test",
        image_size=(512, 512),
        output_size=(512, 512),
        num_classes=7,
        channel_mask=[True, True, True],
        input_strategy=input_strategy,
        output_strategy=output_strategy,
    )

    generator.set_mini_batch_size(2)
    assert generator.mini_batch == 2


@pytest.mark.development
def test_set_mini_batch_size_too_large() -> None:
    patch = MonkeyPatch()
    # mock list directory
    patch.setattr(os, "listdir", lambda x: ["a", "b", "c"])

    input_strategy = DummyStrategy()
    output_strategy = DummyStrategy()

    # create generator instance
    generator = FlowGeneratorExperimental(
        image_path="tests/segmentation_utils_tests/flow_reader_test",
        mask_path="tests/segmentation_utils_tests/flow_reader_test",
        image_size=(512, 512),
        output_size=(512, 512),
        num_classes=7,
        channel_mask=[True, True, True],
        input_strategy=input_strategy,
        output_strategy=output_strategy,
    )
    with pytest.raises(ValueError) as exc_info:
        generator.set_mini_batch_size(5)

    assert (
        exc_info.value.args[0]
        == "The mini batch size cannot be larger than the batch size"
    )


@pytest.mark.development
def test_set_mini_batch_size_not_devisable() -> None:
    patch = MonkeyPatch()
    # mock list directory
    patch.setattr(os, "listdir", lambda x: ["a", "b", "c"])

    input_strategy = DummyStrategy()
    output_strategy = DummyStrategy()

    # create generator instance
    generator = FlowGeneratorExperimental(
        image_path="tests/segmentation_utils_tests/flow_reader_test",
        mask_path="tests/segmentation_utils_tests/flow_reader_test",
        image_size=(512, 512),
        output_size=(512, 512),
        num_classes=7,
        channel_mask=[True, True, True],
        batch_size=3,
        input_strategy=input_strategy,
        output_strategy=output_strategy,
    )
    with pytest.raises(ValueError) as exc_info:
        generator.set_mini_batch_size(2)

    assert (
        exc_info.value.args[0]
        == "The batch size must be divisible by the mini batch size"
    )


@pytest.mark.development
def test_read_batch_get_item() -> None:
    patch = MonkeyPatch()
    # mock list directory
    patch.setattr(os, "listdir", lambda x: ["a", "b", "c"])

    input_strategy = DummyStrategy()
    output_strategy = DummyStrategy(input_shape=(512, 512))

    # create generator instance

    generator = FlowGeneratorExperimental(
        image_path="tests/segmentation_utils_tests/flow_reader_test",
        mask_path="tests/segmentation_utils_tests/flow_reader_test",
        batch_size=2,
        image_size=(512, 512),
        output_size=(512, 512),
        num_classes=7,
        channel_mask=[True, True, True],
        input_strategy=input_strategy,
        output_strategy=output_strategy,
    )

    batch = generator[0]

    assert batch[0].shape == (2, 512, 512, 3)
    assert batch[1].shape == (2, 512, 512, 7)


@pytest.mark.development
def test_read_batch_get_item_expand_dim_fail() -> None:
    with pytest.raises(ValueError) as exc_info:
        patch = MonkeyPatch()
        # mock list directory
        patch.setattr(os, "listdir", lambda x: ["a", "b", "c"])

        input_strategy = DummyStrategy()
        output_strategy = DummyStrategy(input_shape=(512, 512, 1))

        # create generator instance

        generator = FlowGeneratorExperimental(
            image_path="tests/segmentation_utils_tests/flow_reader_test",
            mask_path="tests/segmentation_utils_tests/flow_reader_test",
            batch_size=2,
            image_size=(512, 512),
            output_size=(512, 512),
            num_classes=7,
            channel_mask=[True, True, True],
            input_strategy=input_strategy,
            output_strategy=output_strategy,
        )

        batch = generator[0]


################
# Staging tests#
################


@pytest.mark.staging
def test_read_batch_staging() -> None:
    classes = 7
    n_images = 4
    # prepare test files
    for i in range(n_images):
        image = np.random.randint(0, 255, (512, 512, 3))
        mask = np.random.randint(0, classes, (512, 512))
        np.save(f"tests/segmentation_utils_tests/flow_reader_test/image_{i}", image)
        np.save(f"tests/segmentation_utils_tests/flow_reader_test/mask_{i}", mask)

    dummy_model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                input_shape=(512, 512, 3), filters=3, kernel_size=(3, 3), padding="same"
            ),
            tf.keras.layers.Conv2D(classes, kernel_size=(1, 1), padding="same"),
        ]
    )
    dummy_model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    reader = FlowGeneratorExperimental(
        image_path="tests/segmentation_utils_tests/flow_reader_test",
        mask_path="tests/segmentation_utils_tests/flow_reader_test",
        image_size=(512, 512),
        output_size=(512, 512),
        num_classes=classes,
        channel_mask=[True, True, True],
    )
