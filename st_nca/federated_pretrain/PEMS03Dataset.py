from flautim.pytorch.Dataset import Dataset 
import numpy as np
import torch
from torch.utils.data import DataLoader
import copy

class PEMS03Dataset(Dataset):

    def __init__(self, **kwargs):
        super(PEMS03Dataset, self).__init__("PEMS03", **kwargs)

    def train(self) -> Dataset:
        return copy.deepcopy(self)

    def validation(self) -> Dataset:
        return copy.deepcopy(self) 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), torch.LongTensor([self.labels[idx]])