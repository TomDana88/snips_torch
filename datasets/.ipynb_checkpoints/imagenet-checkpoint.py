import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image

class IMAGENET(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.paths = filter(lambda path: path.lower().endswith('jpg') or path.lower().endswith('jpeg'), os.listdir(root))
        self.paths = list(map(lambda path: os.path.join(root, path), self.paths))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        X = Image.open(path).convert('RGB')
        if self.transform is not None:
            X = self.transform(X)
        return X, 0 # discard label

