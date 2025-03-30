import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim
from os import path
from torch import Tensor
from typing import Iterable, Union
from PIL import Image
from torchvision import transforms
import pickle

from typing import Iterable, Union
_params_t = Union[Iterable[Tensor], Iterable[dict]]

class MGDA_Data:
    """
    This class bundles methods regarding the datasets for MGDA method.

    Example:
        >> X, y = MGDA_Data.get_data(path="")
        >> MGDA_Data.visualize_training_data(X, y)
    """

    @staticmethod
    def get_data(path: str, train_loader, test_loader):
        """
        Returns a list X which contains N 32x32 images and a corresponding list y with the target values.

        :param path: root path, where the folder 'processed' is located
        :return: X and y (both as np.array). X[i] has a shape of (32, 32), y[i] has a shape of (2, 10)
        """

        raw_X = []
        raw_y = []

        # Combine data in train and test_loader
        for dat in train_loader:
            ims = dat[0].numpy()
            ims = [item for sublist in ims for item in sublist]
            raw_X.extend(ims)
            labels = zip(dat[1], dat[2])
            raw_y.extend(labels)

        for dat in test_loader:
            ims = dat[0].numpy()
            ims = [item for sublist in ims for item in sublist]
            raw_X.extend(ims)
            labels = zip(dat[1], dat[2])
            raw_y.extend(labels)

        raw_X = np.array(raw_X)
        raw_y = np.array(raw_y)
        raw_X.reshape(len(raw_X), 28, 28)

        # Enlargen image for better compatibility with CNNs
        X = MGDA_Data.create_bigger_image(raw_X)

        # One hot encode all labels
        y = np.array([[MGDA_Data.one_hot_encode(label[0]), MGDA_Data.one_hot_encode(label[1])]
                      for label in raw_y])
        return X, y

    @staticmethod
    def one_hot_encode(num, length=10):
        return [int(num == index) for index in range(length)]

    @staticmethod
    def one_hot_decode(array):
        return np.where(array == 1)[0][0]

    @staticmethod
    def visualize_training_data(X, y):
        # Create figure w. 9 pictures
        figure = plt.figure(figsize=(8, 8))
        cols, rows = 3, 3

        # Set each figure individually
        for i in range(1, cols * rows + 1):
            # takes random sample (using random sample index)
            sample_idx = torch.randint(len(X), size=(1,)).item()

            # Training data accessed by indexing returns img, label-tuple
            img, label = X[sample_idx], y[sample_idx]

            # Adds/Connects all subplots
            figure.add_subplot(rows, cols, i)

            # Add labels an description
            plt.title(str(label))
            plt.axis("off")
            plt.imshow(torch.tensor(img).squeeze(), cmap="gray")
        plt.show()
    
