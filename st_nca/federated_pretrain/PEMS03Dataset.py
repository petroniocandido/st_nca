from flautim.pytorch.Dataset import Dataset 
import numpy as np
import torch
from torch.utils.data import DataLoader
import copy

class PEMS03Dataset(Dataset):

    def __init__(self, **kwargs):
        super(PEMS03Dataset, self).__init__("PEMS03", **kwargs)

        pems = kwargs.get('pems',None)
        self.client = kwargs.get('client',0)

        self.sensor = int(pems.columns[self.client + 1])

        self.dataset = pems.get_sensor_dataset(self.sensor, dtype=torch.float32, behavior='nondeterministic')

    def train(self) -> Dataset:
        return self.dataset.train()

    def validation(self) -> Dataset:
        return self.dataset.test()

    def __len__(self):
        return len(self.dataset.num_samples)

    def __getitem__(self, idx):
        return self.dataset[idx]