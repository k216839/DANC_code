import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import transforms
from data.multi_mnist_dataloader import MNISTLoader
def load_MultiMnist_data():

    train_transform = torchvision.transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,)),
                                           transforms.Resize((28, 28))])

    test_transform = torchvision.transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,)),
                                           transforms.Resize((28, 28))])
    data = MNISTLoader(batch_size=[256, 100],
                    train_transform=train_transform,
                    test_transform=test_transform,
                    file_path='MTL_dataset/multi_mnist.pickle')
    train_loader, val_loader, test_loader = data.train_dataloader(), data.val_dataloader(), data.test_dataloader()
    print("Data loaded!")
    len_train = len(train_loader.dataset)
    len_val = len(val_loader.dataset)   
    len_test = len(test_loader.dataset)
    print(f"Train dataset size: {len_train}")
    print(f"Validation dataset size: {len_val}")
    print(f"Test dataset size: {len_test}")
    print("Show sample image...")
    # Get the first batch from the train loader
    images, targets = next(iter(train_loader))

    labs_l = targets[0].squeeze()  
    labs_r = targets[1].squeeze()  

    print(f"Image batch shape: {images.shape}")
    print(f"Left label batch shape: {labs_l.shape}")
    print(f"Right label batch shape: {labs_r.shape}")

    img = images[0]
    plt.figure(figsize=(5, 5))
    plt.imshow(img.squeeze(), cmap='gray')
    # plt.title(f"{targets[0][0].item()} & {targets[1][0].item()}")
    plt.axis('off')
    plt.show()

    return train_loader, val_loader, test_loader
if __name__ == '__main__':
    train_loader, val_loader, test_loader = load_MultiMnist_data()