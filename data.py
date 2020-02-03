import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Resize, ToTensor, Normalize
from torchvision.transforms import Compose


class AudioData(Dataset):
    def __init__(self, audata):
        self.data = audata
        self.transform = get_transform(False, True, False, False)

    def __getitem__(self, idx):
        return self.transform(self.data[idx])

    def __len__(self):
        return len(self.data)


def get_transform(resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(CenterCrop(256))
        if resize:
            options.append(Resize((128, 128)))
        if totensor:
            options.append(ToTensor())
        if normalize:
            options.append(Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = Compose(options)
        return transform
