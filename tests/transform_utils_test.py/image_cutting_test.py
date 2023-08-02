import numpy as np
import rasterio
from PIL import Image
from pytest import MonkeyPatch

from utilities.transform_utils.image_cutting import (cut_ims_in_directory,
                                                     image_cut, image_stich)


def test_image_cut() -> None:
    img = np.zeros((512, 512, 3))
    img[-1, -1, 0] = 1
    cut_ims = image_cut(img, (256, 256), num_bands=3)

    assert cut_ims.shape == (4, 256, 256, 3)
    assert cut_ims[0, 0, 0, 0] == 0
    assert cut_ims[-1, -1, -1, 0] == 1


# @pytest.mark.xfail
# def test_image_cut_incorrect_shape_colum_vector() -> None:
#     # does not pass
#     try:
#         img = np.zeros((512))
#         img[-1, -1, 0] = 1
#         image_cut(img, (256, 256), num_bands=3)
#         assert False
#     except ValueError:
#         assert True


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
    except ValueError:
        assert True


def test_image_cut_slack_cut() -> None:
    img = np.zeros((513, 513, 3))
    img[-2, -2, 0] = 1
    cut_ims = image_cut(img, (256, 256), num_bands=3, pad=False)

    assert cut_ims.shape == (4, 256, 256, 3)
    assert cut_ims[0, 0, 0, 0] == 0
    assert cut_ims[-1, -1, -1, 0] == 1


def test_image_cut_slack_cut_exact() -> None:
    img = np.zeros((512, 512, 3))
    img[-2, -2, 0] = 1
    cut_ims = image_cut(img, (256, 256), num_bands=3, pad=False)

    assert cut_ims.shape == (4, 256, 256, 3)
    assert cut_ims[0, 0, 0, 0] == 0
    assert cut_ims[-1, -2, -2, 0] == 1


def test_image_cut_pad() -> None:
    img = np.zeros((511, 511, 3))
    img[-2, -2, 0] = 1
    cut_ims = image_cut(img, (256, 256), num_bands=3)

    assert cut_ims.shape == (4, 256, 256, 3)
    assert cut_ims[0, 0, 0, 0] == 0
    assert cut_ims[-1, -3, -3, 0] == 1


def test_image_cut_pad_exact() -> None:
    img = np.zeros((512, 512, 3))
    img[-2, -2, 0] = 1
    cut_ims = image_cut(img, (256, 256), num_bands=3)

    assert cut_ims.shape == (4, 256, 256, 3)
    assert cut_ims[0, 0, 0, 0] == 0
    assert cut_ims[-1, -2, -2, 0] == 1


def test_image_cut_incorrect_band() -> None:
    try:
        img = np.zeros((512, 512))
        img[-1, -1] = 1
        image_cut(img, (256, 256), num_bands=3)
        assert False
    except ValueError:
        assert True


def test_image_cut_can_add_dimension() -> None:
    img = np.zeros((512, 512))
    img[-1, -1] = 1
    cut_ims = image_cut(img, (256, 256), num_bands=1)

    assert cut_ims.shape == (4, 256, 256, 1)
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

    stiched_img = image_stich(np.array([img1, img2, img3, img4]), 2, 2, (512, 512))

    assert stiched_img.shape == (512, 512, 3)
    assert stiched_img[0, 0, 0] == 0
    assert stiched_img[-1, -1, 0] == 9


def test_cut_ims_in_directory(mocker) -> None:
    patch = MonkeyPatch()

    path_map = {
        1: np.zeros((512, 512)),
        2: np.zeros((512, 512)),
        3: np.zeros((512, 512)),
    }

    mock_reader = mocker.MagicMock(
        spec=rasterio.DatasetReader,
        read=lambda x: path_map[x],
        shape=(512, 512),
        count=3,
    )

    patch.setattr("os.listdir", lambda _: ["1.tiff", "2.tiff", "3.tiff", "4.tiff"])
    patch.setattr(rasterio, "open", lambda x: mock_reader)
    patch.setattr(Image.Image, "save", lambda x, y: None)

    cut_ims_in_directory("test", "test", (256, 256))

    patch.undo()
    patch.undo()
    patch.undo()
