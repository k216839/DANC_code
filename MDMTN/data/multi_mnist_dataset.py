import pickle
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

class MNIST(torch.utils.data.Dataset):
    def __init__(self, file_path, mode: str, transform=None):
        self.transform = transform
        self.mode = mode

        with open(file_path, "rb") as f:
            trainX, trainLabel, testX, test_label = pickle.load(f)

        if mode == 'train' or mode == 'val':
            train_data, val_data, train_label, val_label = train_test_split(trainX, trainLabel, test_size=0.1, random_state=42)
        if mode == 'train':
            self.X = train_data
            self.y = train_label
        elif mode == 'val':
            self.X = val_data
            self.y = val_label
        else:
            self.X = testX
            self.y = test_label

    def __getitem__(self, index):
        img, target = self.X[index], self.y[index]

        img = Image.fromarray(img.astype(np.uint8), mode="L")

        if self.transform is not None:
            img = self.transform(img)
        labs_l = target[0]
        labs_r = target[1]
        return img, (torch.tensor(labs_l).to(torch.long), torch.tensor(labs_r).to(torch.long))

    def __len__(self):
        return len(self.X)