import torch
import pytorch_lightning as pl
import torchvision
from torchvision import transforms
from typing import List
from data.multi_mnist_dataset import MNIST

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

def load_MultiMnist_data():

    train_transform = torchvision.transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,)),
                                           transforms.Resize((28, 28))])

    test_transform = torchvision.transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,)),
                                           transforms.Resize((28, 28))])
    data = MNISTLoader(batch_size=[1024, 1000],
                    train_transform=train_transform,
                    test_transform=test_transform,
                    file_path='MTL_dataset/multi_mnist.pickle')
    # train_loader, val_loader, test_loader = data.train_dataloader(), data.val_dataloader(), data.test_dataloader()
    # train_dataset, val_dataset, test_dataset = data.train_dataset, data.val_dataset, data.test_dataset 
    print("Data loaded!")
    return data