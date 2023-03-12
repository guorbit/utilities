from utilities.segmentation_utils.image_encoder import ImagePreprocessor
import numpy as np

def test_image_onehot_encoder()->None:
    # predifining input variables
    
    n_classes = 2
    batch_size = 1
    image_size = (256, 256)
   
    # creating a mask with 2 classes
    mask = np.zeros((batch_size,image_size[0]//2 * image_size[1]//2))
    mask[:,::2] = 1

    # creating a onehot mask to compare with the output of the function
    onehot_test = np.zeros((batch_size,image_size[0]//2 * image_size[1]//2,n_classes))
    onehot_test[:,::2,1] = 1
    onehot_test[:,1::2,0] = 1
    
    one_hot_image = ImagePreprocessor.onehot_encode(mask,image_size,n_classes)

    assert one_hot_image.shape == (1, image_size[0]//2 * image_size[1]//2, n_classes)
    assert np.array_equal(one_hot_image,onehot_test)

    