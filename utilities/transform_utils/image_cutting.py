"""
This modules contains functions to manipulate images in the form of numpy arrays.
"""

import os
import pathlib
import sys
from typing import Any
import numpy as np
import rasterio
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm


@DeprecationWarning
def image_cut_legacy(
    image: NDArray[Any], cut_dims: tuple[int, int], num_bands: int = 1
) -> NDArray[Any]:
    """
    Takes an input image "image" and cuts into many images each of dimensions "cut_dims".
    Assumed input image shape [CH,CW,C]
        where   H = image height,
                W = image width,
                C = number of channels/bands.
    output image shape [N, H, W, C]
        where   N = as many small images as reqired to fully represent big image.
                H = height of small images specfied in cut_dims[0]
                W = width of small images specfied in cut_dims[1]
                C = Number of channels in input image.
    #!Note, image are cut in row major order.
    #!Note, if image dimensions arent a multiple of cut_dims, then the input image is padded
    #!at the right, and left edge with black pixels. So some images will contain black areas.

    Arguments
    ---------
    :NDArray[Any] image: Numpy array representing image to cut.
    :tuple[int, int] cut_dims: desired dimensions to cut images to.
    :int, optional num_bands: Number of bands of input image. Defaults to 1.

    Returns
    -------
    :return NDArray[Any]: an array of multiple rgb images that represent input images,
    in row major order.
    """

    im_shape = image.shape
    if (len(im_shape) != 2) and (len(im_shape) != 3):
        raise ValueError("input image must be either a matrix, or a 3d tensor")

    if (len(im_shape) == 2) and num_bands != 1:
        raise ValueError(
            "input image is a 2d matrix, but input number of bands is not 1"
        )

    if len(im_shape) == 2:
        image = np.expand_dims(image, axis=-1)

    edge_space = (im_shape[1] % cut_dims[1], im_shape[0] % cut_dims[0])

    num_ims_x = int(np.ceil(im_shape[1] / cut_dims[1]))
    num_ims_y = int(np.ceil(im_shape[0] / cut_dims[0]))
    num_ims = num_ims_x * num_ims_y

    cut_ims = np.zeros((num_ims, cut_dims[0], cut_dims[1], num_bands), dtype=np.int16)

    i = 0

    for y_ims in range(num_ims_y):
        for x_ims in range(num_ims_x):
            if (
                x_ims == num_ims_x - 1
                and y_ims != num_ims_y - 1
                and edge_space != (0, 0)
            ):
                cut_ims[i, :, : edge_space[1], :] = image[
                    y_ims * cut_dims[0] : (y_ims + 1) * cut_dims[0],
                    x_ims * cut_dims[1] :,
                    :,
                ]
            elif (
                x_ims != num_ims_x - 1
                and y_ims == num_ims_y - 1
                and edge_space != (0, 0)
            ):
                cut_ims[i, : edge_space[0], :, :] = image[
                    y_ims * cut_dims[0] :,
                    x_ims * cut_dims[1] : (x_ims + 1) * cut_dims[1],
                    :,
                ]
            elif (
                x_ims == num_ims_x - 1
                and y_ims == num_ims_y - 1
                and edge_space != (0, 0)
            ):
                cut_ims[i, : edge_space[0], : edge_space[1], :] = image[
                    y_ims * cut_dims[0] :, x_ims * cut_dims[1] :, :
                ]
            else:
                cut_ims[i, :, :, :] = image[
                    y_ims * cut_dims[0] : (y_ims + 1) * cut_dims[0],
                    x_ims * cut_dims[1] : (x_ims + 1) * cut_dims[1],
                    :,
                ]
            i += 1

    return cut_ims


def image_cut(
    image: NDArray[Any], cut_dims: tuple[int, int], num_bands: int = 1, pad: bool = True
) -> NDArray[Any]:
    """
    Takes an input image "image" and cuts into many images each of dimensions "cut_dims".
    Assumed input image shape [CH,CW,C]
        where   H = image height,
                W = image width,
                C = number of channels/bands.
    output image shape [N, H, W, C]
        where   N = as many small images as reqired to fully represent big image.
                H = height of small images specfied in cut_dims[0]
                W = width of small images specfied in cut_dims[1]
                C = Number of channels in input image.
    #!Note, image are cut in row major order.
    #!Note, if image dimensions arent a multiple of cut_dims, then the input image is padded
    #!at the right, and left edge with black pixels. So some images will contain black areas.

    Arguments
    ---------
    :NDArray[Any] image: Numpy array representing image to cut.
    :tuple[int, int] cut_dims: desired dimensions to cut images to.
    :int, optional num_bands: Number of bands of input image. Defaults to 1.

    Returns
    -------
    :return NDArray[Any]: an array of multiple rgb images that represent input images,
    in row major order.
    """
    im_shape = image.shape
    if (len(im_shape) != 2) and (len(im_shape) != 3):
        raise ValueError("input image must be either a matrix, or a 3d tensor")

    if (len(im_shape) == 2) and num_bands != 1:
        raise ValueError(
            "input image is a 2d matrix, but input number of bands is not 1"
        )

    if len(im_shape) == 2:
        image = np.expand_dims(image, axis=-1)

    def _get_padded_img(image):
        diff = [0, 0]
        if image.shape[0] % cut_dims[0] != 0:
            diff[0] = cut_dims[0] - image.shape[0] % cut_dims[0]
        if image.shape[1] % cut_dims[1] != 0:
            diff[1] = cut_dims[1] - image.shape[1] % cut_dims[1]

        image = np.pad(image, ((0, diff[0]), (0, diff[1]), (0, 0)), mode="constant")
        return image

    def _cut_image_slack(image):
        # crop image to remove slack
        cut_values = [0, 0, 0, 0]
        if image.shape[0] % cut_dims[0] != 0:
            cut_x = image.shape[0] % cut_dims[0]
            cut_values[0] = cut_x // 2
            cut_values[1] = cut_x - cut_values[0]
        if image.shape[1] % cut_dims[1] != 0:
            cut_y = image.shape[1] % cut_dims[1]
            cut_values[2] = cut_y // 2
            cut_values[3] = cut_y - cut_values[2]

        image = image[
            cut_values[0] : -cut_values[1] or None,
            cut_values[2] : -cut_values[3] or None,
            :,
        ]
        return image

    if pad:
        # calculate padding
        image = _get_padded_img(image)
    else:
        # crop image to remove slack
        image = _cut_image_slack(image)

    img_counts = (image.shape[0] // cut_dims[0], image.shape[1] // cut_dims[1])

    # reshape image to be [N, H, W, C]
    image = image.reshape(
        (img_counts[0], cut_dims[0], img_counts[1], cut_dims[1], num_bands)
    )

    image = image.transpose((0, 2, 1, 3, 4))
    image = image.reshape(
        (img_counts[0] * img_counts[1], cut_dims[0], cut_dims[1], num_bands)
    )

    return image


@DeprecationWarning
def image_stich_legacy(
    ims: NDArray[np.int16], num_ims_x: int, num_ims_y: int, edge_space: tuple[int, int]
) -> NDArray[np.int16]:
    """Stiches input images "ims", into a single returned image.
    assumed layout of input images is [N, H, W, C].
        where   N = Number of images,
                H = individual image height,
                W = indvididual image width,
                C = number of channels/bands.
    return image is off shape [CH, CW, C]
        where   CH = combined height of small images = num_ims_y*H - (H-edges_pace[0])
                CW = combined width of small images = num_ims_x*W - (W-edges_pace[1])
                C = number of channels/bands.
    It is also assumed that the images in Ims are stored in row major order.
    #! Note edge_space = im_dims-black_space. i.e. edge space is the amount of
    #! image that is not black.

    Arguments
    ---------
    :NDArray[np.int16] ims: Images tensor to combine in shape [N, H, W, C].
    :int num_ims_x: Specfies how many small images to stack in the x direction to make big image.
    :int num_ims_y: Specfies how many small images to stack in the y direction to make big image.
    :tuple[int] edge_space: The number of pixels in each direction that are not black space,
    of the most bottom left image.

    Returns
    -------
    :return NDArray[np.int16]: 1 large image as a numpy array in shape [CH, CW, C]
    """
    # assumed layout [b, h, w, c] where b = images to combine,h=height, w=width, c = bands

    ims_shape = ims.shape
    black_space = (ims_shape[1] - edge_space[0], ims_shape[2] - edge_space[1])
    im_dims = (
        num_ims_y * ims_shape[1] - (black_space[0]),
        num_ims_x * ims_shape[2] - black_space[1],
        ims_shape[3],
    )

    image = np.zeros(im_dims, dtype=np.int16)
    im_index = 0
    for y_index in range(num_ims_y):
        for x_index in range(num_ims_x):
            if x_index == num_ims_x - 1 and y_index != num_ims_y - 1:
                image[
                    y_index * ims_shape[1] : (y_index + 1) * ims_shape[1],
                    x_index * ims_shape[2] :,
                    :,
                ] = ims[im_index, :, : edge_space[0], :]
            elif x_index != num_ims_x - 1 and y_index == num_ims_y - 1:
                image[
                    y_index * ims_shape[1] :,
                    x_index * ims_shape[2] : (x_index + 1) * ims_shape[2],
                    :,
                ] = ims[im_index, : edge_space[1], :, :]
            elif x_index == num_ims_x - 1 and y_index == num_ims_y - 1:
                image[y_index * ims_shape[1] :, x_index * ims_shape[2] :, :] = ims[
                    im_index, : edge_space[1], : edge_space[0], :
                ]
            else:
                image[
                    y_index * ims_shape[1] : (y_index + 1) * ims_shape[1],
                    x_index * ims_shape[2] : (x_index + 1) * ims_shape[2],
                    :,
                ] = ims[im_index, :, :, :]
            im_index += 1

    return image


def image_stich(
    ims: NDArray[np.int16], num_ims_x: int, num_ims_y: int, edge_space: tuple[int, int]
) -> NDArray[np.int16]:
    """Stiches input images "ims", into a single returned image.
    assumed layout of input images is [N, H, W, C].
        where   N = Number of images,
                H = individual image height,
                W = indvididual image width,
                C = number of channels/bands.
    return image is off shape [CH, CW, C]
        where   CH = combined height of small images = num_ims_y*H - (H-edges_pace[0])
                CW = combined width of small images = num_ims_x*W - (W-edges_pace[1])
                C = number of channels/bands.
    It is also assumed that the images in Ims are stored in row major order.
    #! Note edge_space = im_dims-black_space. i.e. edge space is the amount of
    #! image that is not black.

    Arguments
    ---------
    :NDArray[np.int16] ims: Images tensor to combine in shape [N, H, W, C].
    :int num_ims_x: Specfies how many small images to stack in the x direction to make big image.
    :int num_ims_y: Specfies how many small images to stack in the y direction to make big image.
    :tuple[int] edge_space: The number of pixels in each direction that are not black space,
    of the most bottom left image.

    Returns
    -------
    :return NDArray[np.int16]: 1 large image as a numpy array in shape [CH, CW, C]
    """
    ims = ims.reshape(ims.shape[1] * num_ims_x, ims.shape[2] * num_ims_y, ims.shape[3])
    black_space = (ims.shape[0] - edge_space[0], ims.shape[1] - edge_space[1])
    padding = (
        black_space[0] // 2,
        black_space[0] // 2 - black_space[0],
        black_space[1] // 2,
        black_space[0] // 2 - black_space[1],
        0,
    )
    ims = ims[padding[0] : -padding[1] or None, padding[2] : -padding[3] or None, :]
    return ims


def preprocess_mask_image(mask_image):
    """
    Preprocesses mask image to be used in training. removes dimensionality from 3 to 2
    and sets values to class indices.

    Arguments
    ---------
    :NDArray[np.int16] mask_image: Mask image to preprocess.

    Returns
    -------
    :return NDArray[np.int16]: Preprocessed mask image.
    """
    mask_image = mask_image / 255

    mask_image = mask_image[:, :, 0] + mask_image[:, :, 1] * 2 + mask_image[:, :, 2] * 4
    mask_image = np.where(mask_image > 0, mask_image - 1, mask_image)
    return mask_image


def cut_ims_in_directory(
    path_ims: str,
    path_target_dir: str,
    target_dims: tuple[int, int] = (512, 512),
    mask=False,
    preprocess: bool = False,
) -> None:
    """Finds images at "Path_ims" cuts them into dimension "target_dims",
    and then saves them as png files to "path_target_dir".
    Note currently only supports rgb images.

    Arguments
    ---------
    :str path_ims: Path to directory where images are stored.
    :str path_target_dir: path to directory where cut images should be placed
    :tuple[int], optional target_dims: target dimensons of image. Defaults to (512,512).
    :bool, optional mask: If true assumes images are masks. Defaults to False.
    :bool, optional preprocess: If true preprocesses images. Defaults to False.
    """
    print("the following files are located at input Path :")
    dir_contents = os.listdir(path_ims)
    dir_contents = sorted(dir_contents)
    print(dir_contents)
    batch_size = 100
    batch = None
    batch_counter = 0
    counter = 0
    channel = 3

    for im_name in tqdm(dir_contents):
        im_path = path_ims + "\\" + im_name
        rasterio_data = rasterio.open(im_path)
        remainder = len(dir_contents) - batch_counter * batch_size
        if remainder < batch_size:
            batch_size = remainder

        # initialize tmp array to store image data
        tmp = np.empty(
            (rasterio_data.shape[0], rasterio_data.shape[1], rasterio_data.count),
            dtype=np.int16,
        )
        tmp[:, :, 0] = rasterio_data.read(1)
        tmp[:, :, 1] = rasterio_data.read(2)
        tmp[:, :, 2] = rasterio_data.read(3)

        # cut image into target dimensions
        cut_im = image_cut(tmp, target_dims, num_bands=3, pad=False)

        # set target channel depending on mask or not
        if mask:
            channel = 1
        else:
            channel = rasterio_data.count

        # initialize batch array
        if batch is None:
            batch = np.empty(
                (
                    batch_size,
                    cut_im.shape[0],
                    cut_im.shape[1],
                    cut_im.shape[2],
                    channel,
                ),
                dtype=np.int8,
            )

        # fill batch array
        for i, n in enumerate(cut_im):
            if preprocess:
                n = preprocess_mask_image(n)
            if mask:
                batch[counter, i, :, :, 0] = n[:, :]
            else:
                batch[counter, i, :, :, 0] = n[:, :, 0]
                batch[counter, i, :, :, 1] = n[:, :, 1]
                batch[counter, i, :, :, 2] = n[:, :, 2]

        # save batch to disk
        if counter == batch_size - 1 or im_name == dir_contents[-1]:
            print(f"Loaded {counter+1} images\nPerforming IO operations...")
            batch = np.reshape(
                batch,
                (batch_size * cut_im.shape[0], target_dims[0], target_dims[1], channel),
            )

            for i, n in enumerate(tqdm(batch)):
                if mask:
                    cut_im = Image.fromarray(np.squeeze(n), mode="L")
                else:
                    cut_im = Image.fromarray(n, mode="RGB")
                cut_im.save(
                    path_target_dir + "\\" + str(batch_counter) + "_" + str(i) + ".png"
                )
            batch = None
            batch_counter += 1
            counter = 0
        else:
            counter += 1


def main():
    seg_load = False
    preprocess = False
    PATH = str(pathlib.Path().resolve())
    ims_path = PATH + "\\ims"
    target_path = PATH + "\\cut_ims"
    cut_ims_in_directory(ims_path, target_path)
    if len(sys.argv) > 1:
        if "--segmentation" in sys.argv:
            seg_load = True
        if "--preprocess" in sys.argv:
            seg_load = True
            preprocess = True
    if seg_load:
        mask_path = PATH + "\\masks"
        target_mask_path = PATH + "\\cut_masks"
        cut_ims_in_directory(
            mask_path, target_mask_path, mask=True, preprocess=preprocess
        )


if __name__ == "__main__":
    main()
