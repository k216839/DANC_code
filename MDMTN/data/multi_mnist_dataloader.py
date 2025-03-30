import torch
import pytorch_lightning as pl
from data.multi_mnist_dataset import MNIST
from typing import List
class MNISTLoader(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: List[int] = [256, 100],
        train_transform=None,
        test_transform=None,
        *args, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = MNIST(mode='train', transform=train_transform, *args, **kwargs)
        self.val_dataset = MNIST(mode='val', transform=test_transform, *args, **kwargs)
        self.test_dataset = MNIST(mode='test', transform=test_transform, *args, **kwargs)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size = self.batch_size[0],
            num_workers = 4,
            shuffle = True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size = self.batch_size[1],
            num_workers = 4,
            shuffle = False
        )
        
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size = self.batch_size[1],
            num_workers = 4,
            shuffle = False
        )
        