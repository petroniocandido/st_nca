import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import copy

import torch
from torch.utils.data import Dataset

from st_nca.embeddings.temporal import TemporalEmbedding, to_pandas_datetime
from st_nca.embeddings.spatial import SpatialEmbedding
from st_nca.embeddings.normalization import ZTransform
from st_nca.tokenizer import NeighborhoodTokenizer

class PEMS03:

    def __init__(self,**kwargs):

      self.dtype = kwargs.get('dtype',torch.float64)
      self.device = kwargs.get('device','cpu')

      edges = pd.read_csv(kwargs.get('edges_file','edges.csv'), engine='pyarrow')
      nodes = pd.read_csv(kwargs.get('nodes_file','nodes.csv'), engine='pyarrow')
      self.data = pd.read_csv(kwargs.get('data_file','data.csv'), engine='pyarrow')
      self.data['timestamp'] = to_pandas_datetime(self.data['timestamp'].values)

      self.ztransform = ZTransform(torch.tensor(self.data[self.data.columns[1:]].values,
                                                dtype=self.dtype, device=self.device),
                                                dtype=self.dtype, device=self.device)

      # Create the graph
      self.G=nx.Graph()
      for row in edges.iterrows():
        self.G.add_edge(int(row[1]['source']),int(row[1]['target']), weight=row[1]['weight'])

      del(edges)

      coordinates = {}

      for ix, node in enumerate(self.G.nodes()):

        _, lat, lon = nodes[nodes['sensor'] == node].values[0]

        coordinates[node] = {'lat': lat, 'lon': lon }

      nx.set_node_attributes(self.G, coordinates)

      self.node_embeddings = SpatialEmbedding(self.G, dtype=self.dtype, device=self.device)

      # The maximum sequence length is equal to the maximum graph degree, or the
      # maximum number of neighbors a node have in the graph
      self.max_length = max([d for n, d in self.G.degree()]) + 1

      # precompute and store all time embeddings to save processing
      self.time_embeddings = TemporalEmbedding(self.data['timestamp'], dtype=self.dtype, device=self.device)

      self.num_sensors = len(nodes)

      del(nodes)

      #self.sensors = sorted([k for k in self.G.nodes()])

      self.num_samples = len(self.data)
      self.token_dim = 7

      self.value_index = 4

      self.tokenizer = NeighborhoodTokenizer(dtype = self.dtype, device = self.device,
                                             graph = self.G, num_nodes = self.num_sensors,
                                             max_length = self.max_length, 
                                             token_dim = self.token_dim, 
                                             ztransform = self.ztransform,
                                             spatial_embedding = self.node_embeddings,
                                             temporal_embedding = self.time_embeddings)

    def get_sample(self, sensor, index):
      #print(sensor, index)      
      X = self.tokenizer.tokenize_sample(self.data, sensor, index)
      y = torch.tensor(self.data[str(sensor)].values[index+1], dtype=self.dtype, device=self.device)
      return X,y

    # Will returna a SensorDataset filled with the sensor & neighbors preprocessed data (X)
    # and the expected values for t+y (y)
    def get_sensor_dataset(self, sensor, train = 0.7, dtype = torch.float64, **kwargs):
      X = self.tokenizer.tokenize_all(self.data, sensor)[:-1]
      y = torch.tensor(self.data[str(sensor)].values[1:], dtype=self.dtype, device=self.device)
      return SensorDataset(str(sensor),X,y,train, dtype, num_features = self.num_sensors,
                           max_length=self.max_length, token_dim=self.token_dim,
                           value_index=self.value_index, **kwargs)

    def get_fewsensors_dataset(self, sensors, train = 0.7, dtype = torch.float64, **kwargs):
      X = None
      y = None
      try:
        for sensor in sensors:
          tmpX = self.tokenizer.tokenize_all(self.data, sensor)[:-1]
          tmpy = torch.tensor(self.data[str(sensor)].values[1:], dtype=self.dtype, device=self.device)
          if X is None:
            X = tmpX
            y = tmpy
          else:
            #X = np.vstack((X,tmpX))
            X = torch.vstack((X,tmpX))
            #y = np.hstack((y,tmpy))
            y = torch.hstack((y,tmpy))
      except Exception as ex:
        print(sensor, str(ex))

      return SensorDataset('FEW',X,y,train, dtype, num_features = self.num_sensors,
                           max_length=self.max_length, token_dim=self.token_dim,
                           value_index=self.value_index, **kwargs)

    
    def get_breadth_dataset(self, start_sensor, max_sensors = 20, train = 0.7, dtype = torch.float64, **kwargs):
      sensors = []
      next = [start_sensor]
      m = 0
      while m < max_sensors:
        for sensor in next:
          if sensor not in sensors: 
            sensors.append(sensor)
            m += 1
            next.remove(sensor)
            if m < max_sensors:
              for neighbor in self.G.neighbors(sensor):
                next.append(neighbor)
            else:
              break

      return self.get_fewsensors_dataset(sensors, train = train, dtype = dtype, **kwargs), sensors

    def get_allsensors_dataset(self, **kwargs):
      return AllSensorDataset(pems=self, **kwargs)
    
    def get_sensor(self, index):
      return int(self.data.columns[index + 1])
    
    def to(self, *args, **kwargs):
      if isinstance(args[0], str):
        self.device = args[0]
      else:
        self.dtype = args[0]
      return self
    

def self_supervised_transform(x, pems):
  r = np.random.rand()
  if r >= .9:
    # Remove the cell value and keep all the neighbors
    x[0,:] = torch.full([pems.token_dim], pems.NULL_SYMBOL, dtype = pems.dtype, device=pems.device)
  elif r >= .8:
    # remove the cell value e remove all neighbor values
    x[0,pems.value_index] = torch.tensor([pems.NULL_SYMBOL], dtype = pems.dtype, device=pems.device)
    x[1:,:] = torch.full((pems.max_length-1, pems.token_dim),pems.NULL_SYMBOL, dtype = pems.dtype, device=pems.device)
  elif r >= .7:
    # Remove the cell value
    x[0,pems.value_index] = torch.tensor([pems.NULL_SYMBOL], dtype = pems.dtype, device=pems.device)
  elif r >= .6:
    # Remove neighbor values
    x[1:,:] = torch.full((pems.max_length-1, pems.token_dim),pems.NULL_SYMBOL, dtype = pems.dtype, device=pems.device)
  elif r >= .5:
    # Introduce random noise
    x[:,pems.value_index] = x[:,pems.value_index] + torch.randn(pems.max_length, dtype = pems.dtype, device=pems.device)/12
  return x
  

class SensorDataset(Dataset):
  def __init__(self, name, X, y, train = 0.7,
               dtype = torch.float64, **kwargs):
    super().__init__()

    self.NULL_SYMBOL = 0

    self.behavior = kwargs.get('behavior','deterministic')

    self.num_samples = len(X)

    self.num_features = kwargs.get('num_features',0)

    self.max_length = kwargs.get('max_length',0)

    self.token_dim = kwargs.get('token_dim',0)

    self.value_index= kwargs.get('value_index',0)

    self.name = name
    self.dtype = dtype
    self.device = kwargs.get('device','cpu')

    self.X = X.to(self.dtype).to(self.device)
    self.y = y.to(self.dtype).to(self.device)

    self.indexes = torch.randperm(self.num_samples)

    self.X = self.X[self.indexes]
    self.y = self.y[self.indexes]

    self.train_split = int(train * self.num_samples)
    self.is_validation = False

  def train(self) -> Dataset:
    tmp = copy.deepcopy(self)
    tmp.is_validation = False
    tmp.X = self.X[:self.train_split].to(self.device)
    tmp.y = self.y[:self.train_split].to(self.device)
    tmp.num_samples = tmp.X.size(0)
    return tmp

  def test(self) -> Dataset:
    tmp = copy.deepcopy(self)
    tmp.is_validation = True
    tmp.X = self.X[self.train_split:].to(self.device)
    tmp.y = self.y[self.train_split:].to(self.device)
    tmp.num_samples = tmp.X.size(0)
    return tmp

  # Make this function stochastic!
  # Sometimes X will return the full information,sometimes X will miss some part of the information,
  # and sometimes will return with corrupted information (add noise)
  def __getitem__(self, index):
    if self.behavior == 'deterministic':
      return self.X[index], self.y[index]
    else:
      x = torch.clone(self.X[index])
      x = self_supervised_transform(x, self)
      return x, self.y[index]

  def __len__(self):
    return self.num_samples

  def __iter__(self):
    for ix in range(self.num_samples):
      yield self[ix]

  def __str__(self):
    return "Dataset {}: {} attributes {} samples".format(self.name, self.num_attributes, self.num_samples)
  
  def to(self, *args, **kwargs):
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.X = self.X.to(*args, **kwargs)
    self.y = self.y.to(*args, **kwargs)
    return self
  

class AllSensorDataset(Dataset):
  def __init__(self, pems, train = 0.7, **kwargs):
    super().__init__()

    self.pems = pems

    self.max_length = pems.max_length

    self.token_dim = pems.token_dim

    self.behavior = kwargs.get('behavior','deterministic')

    self.train_pct = train

    self.train_split = int(train * self.pems.num_samples) 
    self.test_split = self.pems.num_samples - self.train_split 

    self.samples = self.pems.num_samples * self.pems.num_sensors

    self.is_validation = False

  def train(self) -> Dataset:
    tmp = copy.deepcopy(self)
    tmp.is_validation = False
    return tmp

  def test(self) -> Dataset:
    tmp = copy.deepcopy(self)
    tmp.is_validation = True
    return tmp

  def __getitem__(self, index):
    if not self.is_validation:
      train_sensor_ix = index // (self.train_split - 1)
      train_data_ix = index % (self.train_split - 1)
      sensor = self.pems.get_sensor(train_sensor_ix)
      X,y = self.pems.get_sample(sensor, train_data_ix)
    else:
      train_sensor_ix = index // (self.test_split - 1)
      train_data_ix = (index % (self.test_split - 1)) + self.train_split
      sensor = self.pems.get_sensor(train_sensor_ix)
      X,y = self.pems.get_sample(sensor, train_data_ix)
    
    if self.behavior == 'deterministic':
      return X,y 
    else:
      return self_supervised_transform(X, self.pems), y    
    
  def __len__(self):
    if not self.is_validation:
      return (self.train_split - 1) * self.pems.num_sensors 
    else:
      return (self.test_split - 1) * self.pems.num_sensors 

  def __iter__(self):
    for ix in range(self.samples):
      yield self[ix]

  def to(self, *args, **kwargs):
    self.pems = self.pems.to(*args, **kwargs)
    return self
