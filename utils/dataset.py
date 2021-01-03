import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset

from sklearn.model_selection import train_test_split

def train_val_split(dataset, val_ratio=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_ratio)
    datasets = {
        'train': Subset(dataset, train_idx),
        'val': Subset(dataset, val_idx)
    }
    return datasets

def get_datasets(path, transform):
    data = datasets.ImageFolder(path, transform=transform)
    return train_val_split(data)

def get_dataloader(data, batch_size, shuffle, num_workers):
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

