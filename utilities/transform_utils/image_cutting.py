"""
This modules contains functions to manipulate images in the form of numpy arrays.
"""

import os
import pathlib
from typing import Any
import numpy as np
from numpy.typing import NDArray
import rasterio
from PIL import Image


def image_cut(
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


def image_cut_experimental(
    image: NDArray[Any], cut_dims: tuple[int, int], num_bands: int = 1, pad: bool = True
) -> NDArray[Any]:
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

    print(image.shape)

    img_counts = (image.shape[0] // cut_dims[0], image.shape[1] // cut_dims[1])

    # reshape image to be [N, H, W, C]
    image = image.reshape(
        (img_counts[0], cut_dims[0], img_counts[1], cut_dims[1], num_bands)
    )

    image = image.transpose((0, 2, 1, 3, 4))
    image = image.reshape((img_counts[0] * img_counts[1], cut_dims[0], cut_dims[1], 3))

    return image


def image_stich(
    ims: NDArray[np.int16], num_ims_x: int, num_ims_y: int, edge_space: tuple[int]
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


def image_stich_experimental(
    ims: NDArray[np.int16], num_ims_x: int, num_ims_y: int, edge_space: tuple[int]
) -> NDArray[np.int16]:

    # no idea what edge space is 😥😥 so its not implemented yet
    ims = ims.reshape(ims.shape[1] * num_ims_x, ims.shape[2] * num_ims_y, ims.shape[3])

    return ims


def cut_ims_in_directory(
    path_ims: str, path_target_dir: str, target_dims: tuple[int] = (512, 512)
) -> None:
    """Finds images at "Path_ims" cuts them into dimension "target_dims",
    and then saves them as png files to "path_target_dir".
    Note currently only supports rgb images.

    Arguments
    ---------
    :str path_ims: Path to directory where images are stored.
    :str path_target_dir: path to directory where cut images should be placed
    :tuple[int], optional target_dims: target dimensons of image. Defaults to (512,512).
    """
    print("the following files are located at input Path :")
    dir_contents = os.listdir(path_ims)
    print(dir_contents)
    for im_name in dir_contents:

        im_path = path_ims + "\\" + im_name
        rasterio_data = rasterio.open(im_path)
        print(type(rasterio_data))
        image = np.empty(
            (rasterio_data.shape[0], rasterio_data.shape[1], rasterio_data.count),
            dtype=np.int16,
        )
        print(image.shape)
        image[:, :, 0] = rasterio_data.read(1)
        image[:, :, 1] = rasterio_data.read(2)
        image[:, :, 2] = rasterio_data.read(3)

        print(image.shape)

        ims = image_cut(image, target_dims, num_bands=3)
        ims = np.array(ims, dtype=np.int8)

        for index, cut_array in enumerate(ims):

            cut_im = Image.fromarray(cut_array, mode="RGB")
            print(type(cut_im))
            cut_im.save(path_target_dir + "\\" + im_name[0:-4] + str(index) + ".png")


def main():
    PATH = str(pathlib.Path().resolve())
    ims_path = PATH + "\\ims"
    target_path = PATH + "\\cut_ims"

    cut_ims_in_directory(ims_path, target_path)


if __name__ == "__main__":
    main()
