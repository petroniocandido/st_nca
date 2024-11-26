from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import networkx as nx

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from tensordict import TensorDict

from st_nca.cellmodel import CellModel
from st_nca.tokenizer import NeighborhoodTokenizer


class FineTunningDataset(Dataset):
  def __init__(self, pems, sensors, **kwargs):
    super().__init__()

    self.pems = pems
    self.sensors = sensors
    self.num_nodes = len(sensors)
    self.increment_type = kwargs.get('increment_type','minute')
    self.increment = kwargs.get('increment',1)
    self.steps_ahead = kwargs.get('steps_ahead',10)

  def __getitem__(self, date):
    if isinstance(date, datetime):
      dt1 = from_datetime_to_np(date)
      dt2 = from_datetime_to_np(get_timestamp(date, self.increment_type, self.increment))
    elif isinstance(date, np.datetime64):
      dt1 = date
      dt2 = from_datetime_to_np(get_timestamp(from_np_to_datetime(date), self.increment_type, self.increment))

    X = {'timestamp': dt1}
    df1 = self.pems.data[(self.pems.data['timestamp'] == dt1)]
    for ix, node in enumerate(self.sensors):
      X[str(node)] = df1[str(node)].values[0]

    y = torch.zeros(self.num_nodes * self.steps_ahead, dtype=self.pems.dtype, device=self.pems.device)

    for ct in range(0,self.steps_ahead):
      dt2 = from_datetime_to_np(get_timestamp(from_np_to_datetime(date), self.increment_type, self.increment))
      df2 = self.pems.data[(self.pems.data['timestamp'] == dt2)]

      for ix, node in enumerate(self.sensors):
        y[ct * self.num_nodes + ix] = df2[str(node)].values[0]

      date = from_np_to_datetime(dt2)
     
    return X,y

  def __len__(self):
    return self.pems.num_samples

  def __iter__(self):
    for date in self.pems.data['timestamp']:
      yield self[date]

  def to(self, *args, **kwargs):
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.pems = self.pems.to(*args, **kwargs)
    return self