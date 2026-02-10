from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import numpy as np
import torch

from typing import Tuple

BATCH_SIZE    = 10_000
CIFAR_10_SIZE = 5 * BATCH_SIZE
META          = r"Z:\code\cifar10-vae\dataset\batches.meta"
TRAIN         = r"Z:\code\cifar10-vae\dataset\train"
TEST          = r"Z:\code\cifar10-vae\dataset\test"

def unpickle(file: str) -> dict:
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class DataReader(Dataset):
    def __init__(self, path: str, transforms: T.Compose, load=1.0):
        super().__init__()
        
        self.paths  = []
        self.images = []
        self.labels = []
        self.max_load = round(load * (CIFAR_10_SIZE if "train" in path else BATCH_SIZE))
        self.str_labels = unpickle(META)[b"label_names"]
        self.transforms = transforms

        for p in os.listdir(path):
            self.paths.append(os.path.join(path, p))

        reached_limit = False
        for batch in self.paths:
            data = unpickle(batch)
            for img, label in zip(data[b"data"], data[b"labels"]):
                if len(self.images) >= self.max_load:
                    reached_limit = True
                    break
                self.images.append(np.reshape(img, (3, 32, 32)).transpose((1, 2, 0)))
                self.labels.append(label)
            if reached_limit:
                break
            
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img   = self.images[idx]
        label = self.labels[idx]

        img   = self.transforms(img)
        label = torch.tensor(label)

        return img, label

    def __len__(self):
        return len(self.images)