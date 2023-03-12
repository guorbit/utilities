import numpy as np
import tensorflow as tf


class ImagePreprocessor:
    @classmethod
    def onehot_encode(self, masks, image_size, num_classes):
        """
        Onehot encodes the images coming from the image generator object

        Parameters:
        ----------
        masks (tf tensor): masks to be onehot encoded

        Returns:
        -------
        encoded (tf tensor): onehot encoded masks
        """
        encoded = np.zeros(
            (masks.shape[0], image_size[0] // 2 * image_size[1] // 2, num_classes)
        )
        for i in range(num_classes):
            encoded[:, :, i] = tf.squeeze((masks == i).astype(int))

        return encoded

    @classmethod
    def augmentation_pipeline(self, image, mask, input_size, channels=3):
        """
        Applies augmentation pipeline to the image and mask

        Parameters:
        ----------
        image (tf tensor): image to be augmented
        mask (tf tensor): mask to be augmented
        input_size (tuple): size of the input image

        Returns:
        -------
        image (tf tensor): augmented image
        mask (tf tensor): augmented mask
        """

        input_size = (input_size[0], input_size[1], channels)

        seed = np.random.randint(0, 100000)

        image = tf.image.random_flip_left_right(image, seed=seed)
        image = tf.image.random_flip_up_down(image, seed=seed)
        image = tf.image.random_brightness(image, 0.2, seed=seed)
        image = tf.image.random_contrast(image, 0.8, 1.2, seed=seed)
        image = tf.image.random_saturation(image, 0.8, 1.2, seed=seed)
        image = tf.image.random_hue(image, 0.2, seed=seed)


        mask = tf.image.random_flip_left_right(mask, seed=seed)
        mask = tf.image.random_flip_up_down(mask, seed=seed)


        return image, mask

    @classmethod
    def flatten(self, image, input_size, channels=1):
        return tf.reshape(image, (input_size[0] * input_size[1], channels))

