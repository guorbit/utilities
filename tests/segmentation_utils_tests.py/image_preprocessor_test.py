import numpy as np
import pytest
import tensorflow as tf

from utilities.segmentation_utils import ImagePreprocessor


@pytest.mark.skip(reason="Deprecated functionality")
def test_image_onehot_encoder_column() -> None:
    # predifining input variables
    n_classes = 2
    batch_size = 1
    image_size = (512, 512)
    output_size = (256, 256)

    # creating a mask with 2 classes
    mask = np.zeros((batch_size, output_size[0] * output_size[1]))
    mask[:, ::2] = 1

    # creating a onehot mask to compare with the output of the function
    onehot_test = np.zeros((batch_size, output_size[0] * output_size[1], n_classes))
    onehot_test[:, ::2, 1] = 1
    onehot_test[:, 1::2, 0] = 1

    one_hot_image = ImagePreprocessor.onehot_encode(mask, output_size, n_classes)

    assert one_hot_image.shape == (
        1,
        output_size[0] * output_size[1],
        n_classes,
    )
    assert np.array_equal(one_hot_image, onehot_test)

@pytest.mark.development
def test_image_onehot_encoder_squarematrix() -> None:
    # predifining input variables
    n_classes = 2
    batch_size = 1
    image_size = (512, 512)
    output_size = (256, 256)

    # creating a mask with 2 classes
    mask = np.zeros((batch_size, output_size[0], output_size[1]))
    mask[:, ::2,:] = 1

    # creating a onehot mask to compare with the output of the function
    onehot_test = np.zeros((batch_size, output_size[0] , output_size[1], n_classes))
    onehot_test[:, ::2, :,1] = 1
    onehot_test[:, 1::2,:, 0] = 1

    one_hot_image = ImagePreprocessor.onehot_encode(mask, n_classes)

    assert one_hot_image.shape == (
        1,
        output_size[0],
        output_size[1],
        n_classes,
    )
    assert np.array_equal(one_hot_image, onehot_test)

@pytest.mark.development
def test_image_augmentation_pipeline_squarematrix() -> None:
    # predifining input variables
    image = np.zeros((512, 512, 3))
    mask = np.zeros((256, 256, 1))
    image = tf.convert_to_tensor(image)
    mask = tf.convert_to_tensor(mask)

    input_size = (512, 512)
    output_size = (256, 256)

    # creating dummy queues
    image_queue = ImagePreprocessor.PreprocessingQueue(
        queue=[ImagePreprocessor.PreFunction(lambda x, y, seed: x,y=1)]
    )
    mask_queue = ImagePreprocessor.PreprocessingQueue(
        queue=[ImagePreprocessor.PreFunction(lambda x, y, seed: x,y=1)]
    )

    image_new, mask_new = ImagePreprocessor.augmentation_pipeline(
        image,
        mask,
        image_queue=image_queue,
        mask_queue=mask_queue,
    )
    image_new = image_new.numpy()
    mask_new = mask_new.numpy()

    assert image_new.shape == (512, 512, 3)
    assert mask_new.shape == (256, 256, 1)

@pytest.mark.development
def test_processing_queue() -> None:
    # creating dummy queues
    
    image_queue = ImagePreprocessor.PreprocessingQueue(
        queue=[ImagePreprocessor.PreFunction(lambda seed:seed, seed=1)]
    )
    # changing the seed
    new_seed = 5
    image_queue.update_seed(new_seed)

    assert image_queue.queue[0].kwargs["seed"] == new_seed

@pytest.mark.development
def test_generate_default_queue() -> None:
    # creating default queues
    image_queue, mask_queue = ImagePreprocessor.generate_default_queue()

    
    assert image_queue.get_queue_length() == 5
    assert mask_queue.get_queue_length() == 2

@pytest.mark.development
def test_flatten() -> None:
    image = np.zeros((512, 512, 3))
    image = tf.convert_to_tensor(image)
    image = ImagePreprocessor.flatten(image, (512, 512), 3)
    image = image.numpy()
    assert image.shape == (512 * 512, 3)
