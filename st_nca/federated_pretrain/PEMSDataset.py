from flautim.pytorch.Dataset import Dataset 
import torch

class PEMSDataset(Dataset):

    def __init__(self, **kwargs):
        super(PEMSDataset, self).__init__("PEMS", **kwargs)

        pems = kwargs.get('pems',None)
        self.client = kwargs.get('client',0)

        self.type = kwargs.get('type','federated')

        self.sensor = pems.get_sensor(self.client)

        behavior=kwargs.get('behavior','deterministic')

        if self.type == 'federated':
           self.dataset = pems.get_sensor_dataset(self.sensor, dtype=torch.float32, 
                                                  train = self.train_split,
                                                  behavior=behavior)

        elif self.type == 'breadth':
            neighs = kwargs.get('neighbors',20)
            self.dataset, _ = pems.get_breadth_dataset(self.sensor, max_sensors = neighs, 
                                                       dtype=torch.float32, 
                                                       behavior=behavior)
         
        else:
            self.dataset = pems.get_allsensors_dataset(dtype=torch.float32, 
                                                       train = self.train_split,
                                                       behavior=behavior)

    def train(self) -> Dataset:
        return self.dataset.train()

    def validation(self) -> Dataset:
        return self.dataset.test()

    def __len__(self):
        return len(self.dataset.num_samples)

    def __getitem__(self, idx):
        return self.dataset[idx]