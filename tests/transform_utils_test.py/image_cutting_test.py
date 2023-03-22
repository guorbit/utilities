import numpy as np
from utilities.transform_utils.image_cutting import (
    image_cut,
    image_stich,
    image_cut_experimental,
)
import pytest


def test_image_cut() -> None:
    img = np.zeros((512, 512, 3))
    img[-1, -1, 0] = 1
    cut_ims = image_cut(img, (256, 256), num_bands=3)

    assert cut_ims.shape == (4, 256, 256, 3)
    assert cut_ims[0, 0, 0, 0] == 0
    assert cut_ims[-1, -1, -1, 0] == 1


@pytest.mark.xfail
def test_image_cut_incorrect_shape_colum_vector() -> None:
    # does not pass
    try:
        img = np.zeros((512))
        img[-1, -1, 0] = 1
        image_cut(img, (256, 256), num_bands=3)
        assert False
    except ValueError:
        assert True


def test_image_cut_incorrect_shape_too_many() -> None:
    # does not pass
    try:
        img = np.zeros((512, 512, 3, 3))
        img[-1, -1, 0] = 1
        image_cut(img, (256, 256), num_bands=3)
        assert False
    except ValueError:
        assert True


def test_image_cut_incorrect_band_specified() -> None:
    # passes however the function doesn't rasie a value error
    # when the bands do not match
    try:
        img = np.zeros((512, 512, 5))
        img[-1, -1, 0] = 1
        image_cut(img, (256, 256), num_bands=3)
        assert False
    except ValueError as e:
        
        assert True


def test_image_cut_slack_cut() -> None:
    img = np.zeros((513, 513, 3))
    img[-2, -2, 0] = 1
    cut_ims = image_cut_experimental(img, (256, 256), num_bands=3, pad=False)

    assert cut_ims.shape == (4, 256, 256, 3)
    assert cut_ims[0, 0, 0, 0] == 0
    assert cut_ims[-1, -1, -1, 0] == 1


def test_image_stich() -> None:
    img1 = np.zeros((256, 256, 3), dtype=np.uint8)
    img2 = np.zeros((256, 256, 3), dtype=np.uint8)
    img3 = np.zeros((256, 256, 3), dtype=np.uint8)
    img4 = np.zeros((256, 256, 3), dtype=np.uint8)

    for i in range(3):
        img1[:, :, i] = 0 + i
        img2[:, :, i] = 3 + i
        img3[:, :, i] = 6 + i
        img4[:, :, i] = 9 + i

    stiched_img = image_stich(np.array([img1, img2, img3, img4]), 2, 2, (256, 256))

    assert stiched_img.shape == (512, 512, 3)
    assert stiched_img[0, 0, 0] == 0
    assert stiched_img[-1, -1, 0] == 9
