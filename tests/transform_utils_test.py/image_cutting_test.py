import numpy as np
from utilities.transform_utils.image_cutting import image_cut, image_stich



def test_image_cut() -> None:
    img = np.zeros((513, 513, 3))
    cut_ims = image_cut(img, (256, 256), num_bands = 3)

    pass



def test_image_stich() -> None:
    img1 = np.zeros((256, 256, 3), dtype=np.uint8)
    img2 = np.zeros((256, 256, 3), dtype=np.uint8)
    img3 = np.zeros((256, 256, 3), dtype=np.uint8)
    img4 = np.zeros((256, 256, 3), dtype=np.uint8)

    img1[:, :, :] = 1
    img2[:, :, :] = 2
    img3[:, :, :] = 3
    img4[:, :, :] = 4

    stiched_img = image_stich(np.array([img1, img2, img3, img4]), 2,2 , (256,256))

    print(stiched_img.shape)
    assert stiched_img.shape == (512, 512, 3)
    assert stiched_img[0, 0, 0] == 1
    assert stiched_img[-1, -1, 0] == 4
