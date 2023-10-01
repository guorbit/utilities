import os

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm


class S3DatasetOptimizer:
    """
    Builds image batches from a dataset providing optimised IO performance
    :batch_size int: batch size to group images in
    :path string: path to the dataset folder
    :return tuple(list(string),list(string)): tuple of filenames for in and out
    """

    def __init__(self, batch_size: int, path: str):
        self.batch_size = batch_size
        self.path = path
        self.x_names, self.y_names = self.__build_dataset(batch_size, path)

    def __build_dataset(self, batch_size: int, path: str):
        """
        Build dataset from the given path
        :batch_size int: batch size to group images in
        :path string: path to the dataset folder
        :return tuple(list(string),list(string)): tuple of filenames for in and out
        """

        X = sorted(os.listdir(os.path.join(path, "image")))
        y = sorted(os.listdir(os.path.join(path, "label")))
        for X_val, y_val in tqdm(zip(X, y), total=len(X)):
            if X_val.split(".")[0] != y_val.split(".")[0]:
                raise ValueError("Image and label names do not match")

        n_batch = len(X) // batch_size
        diff = len(X) % batch_size

        X_batches = np.array_split(X[:-diff], n_batch)
        y_batches = np.array_split(y[:-diff], n_batch)

        # add diff number of unknown to last batch
        X_last = X[-diff:]
        y_last = y[-diff:]

        for _ in range(batch_size - diff):
            X_last.append("#")
            y_last.append("#")

        X_last = np.array(X_last)
        y_last = np.array(y_last)

        X_batches.append(X_last)
        y_batches.append(y_last)

        # append last not ful batch to end of list

        return X_batches, y_batches

    def export_dataset(
        self,
        output_path: str,
        dtype=np.int8,
    ):
        """
        Export dataset in batches to the given path

        Parameters
        ----------
        :input_path string: path to the dataset folder
        :output_path string: path to the output folder
        :batch_size int: batch size to group images in

        Keyword Arguments
        -----------------
        :dtype numpy.dtype: data type of the images, defaults to np.int8
        """
        number_of_images_exported = 0
        batch_size_ex = self.batch_size

        for i, (X, y) in tqdm(enumerate(zip(self.x_names, self.y_names)), total=len(self.x_names)):
            X_batch = np.zeros((batch_size_ex, 4, 512, 512), dtype=dtype)
            y_batch = np.zeros((batch_size_ex, 1, 512, 512), dtype=dtype)

            for j, (X_image_name, y_image_name) in enumerate(zip(X, y)):
                if X_image_name == "#" or y_image_name == "#":
                    continue
                X_input_path = os.path.join(self.path, "image", X_image_name)
                y_input_path = os.path.join(self.path, "label", y_image_name)

                with rasterio.open(X_input_path) as X_image:
                    X_batch[j, ...] = X_image.read()
                with rasterio.open(y_input_path) as y_image:
                    y_batch[j, ...] = y_image.read()
                number_of_images_exported += 1
            X_output_path = os.path.join(output_path, "image", str(i))
            y_output_path = os.path.join(output_path, "label", str(i))

            X_batch = np.moveaxis(X_batch, 1, -1)
            y_batch = np.moveaxis(y_batch, 1, -1)

            np.save(X_output_path, X_batch)
            np.save(y_output_path, y_batch)
        df = pd.DataFrame([number_of_images_exported, batch_size_ex])
        df.to_csv(os.path.join(output_path, "info.csv"))
