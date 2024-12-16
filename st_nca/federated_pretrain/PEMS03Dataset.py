from flautim.pytorch.Dataset import Dataset 
import torch

class PEMS03Dataset(Dataset):

    def __init__(self, **kwargs):
        super(PEMS03Dataset, self).__init__("PEMS03", **kwargs)

        pems = kwargs.get('pems',None)
        self.client = kwargs.get('client',0)

        self.sensor = pems.get_sensor(self.client)

        self.dataset = pems.get_sensor_dataset(self.sensor, dtype=torch.float32, behavior='deterministic')

    def train(self) -> Dataset:
        return self.dataset.train()

    def validation(self) -> Dataset:
        return self.dataset.test()

    def __len__(self):
        return len(self.dataset.num_samples)

    def __getitem__(self, idx):
        return self.dataset[idx]