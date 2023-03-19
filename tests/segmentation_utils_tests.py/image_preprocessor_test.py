import numpy as np
import tensorflow as tf
from utilities.segmentation_utils import ImagePreprocessor



def test_image_onehot_encoder() -> None:
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
        image_size[0] // 2 * image_size[1] // 2,
        n_classes,
    )
    assert np.array_equal(one_hot_image, onehot_test)


def test_image_augmentation_pipeline_column() -> None:
    # predifining input variables
    image = np.zeros((512, 512, 3))
    mask = np.zeros((256 * 256, 1))
    image = tf.convert_to_tensor(image)
    mask = tf.convert_to_tensor(mask)

    input_size = (512, 512)
    output_size = (256 * 256, 1)
    output_reshape = (256, 256)

    # createing dummy queues
    image_queue = ImagePreprocessor.PreprocessingQueue(
        queue=[lambda x, y, seed: x], arguments=[{"y": 1}]
    )
    mask_queue = ImagePreprocessor.PreprocessingQueue(
        queue=[lambda x, y, seed: x], arguments=[{"y": 1}]
    )

    image_new, mask_new = ImagePreprocessor.augmentation_pipeline(
        image, mask, input_size, output_size, output_reshape, image_queue, mask_queue
    )
    image_new = image_new.numpy()
    mask_new = mask_new.numpy()

    assert np.array(image_new).shape == (512, 512, 3)
    assert np.array(mask_new).shape == (256 * 256, 1, 1)


def test_image_augmentation_pipeline_squarematrix() -> None:
    # predifining input variables
    image = np.zeros((512, 512, 3))
    mask = np.zeros((256, 256, 1))
    image = tf.convert_to_tensor(image)
    mask = tf.convert_to_tensor(mask)

    input_size = (512, 512)
    output_size = (256, 256)

    # createing dummy queues
    image_queue = ImagePreprocessor.PreprocessingQueue(
        queue=[lambda x, y, seed: x], arguments=[{"y": 1}]
    )
    mask_queue = ImagePreprocessor.PreprocessingQueue(
        queue=[lambda x, y, seed: x], arguments=[{"y": 1}]
    )

    image_new, mask_new = ImagePreprocessor.augmentation_pipeline(
        image,
        mask,
        input_size,
        output_size,
        image_queue=image_queue,
        mask_queue=mask_queue,
    )
    image_new = image_new.numpy()
    mask_new = mask_new.numpy()

    assert image_new.shape == (512, 512, 3)
    assert mask_new.shape == (256, 256, 1)


def test_processing_queue() -> None:
    # createing dummy queues
    image_queue = ImagePreprocessor.PreprocessingQueue(
        queue=[lambda seed: seed], arguments=[dict(seed=1)]
    )

    # changing the seed
    new_seed = 5
    image_queue.update_seed(new_seed)

    assert image_queue.arguments[0]["seed"] == new_seed


def test_generate_default_queue() -> None:
    # createing default queues
    image_queue, mask_queue = ImagePreprocessor.generate_default_queue()

    # changing the seed
    new_seed = 5
    image_queue.update_seed(new_seed)

    assert image_queue.arguments[0]["seed"] == new_seed
    assert image_queue.get_queue_length() == 6
    assert mask_queue.get_queue_length() == 2


def test_flatten() -> None:
    image = np.zeros((512, 512, 3))
    image = tf.convert_to_tensor(image)
    image = ImagePreprocessor.flatten(image, (512, 512), 3)
    image = image.numpy()
    assert image.shape == (512 * 512, 1, 3)

