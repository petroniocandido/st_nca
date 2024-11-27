from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import copy
import time

from sklearn.manifold import SpectralEmbedding

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from tensordict import TensorDict

from st_nca.embeddings.temporal import TemporalEmbedding, to_pandas_datetime
from st_nca.embeddings.spatial import SpatialEmbedding
from st_nca.embeddings.normalization import ZTransform
from st_nca.tokenizer import NeighborhoodTokenizer


class PEMS03:

    def __init__(self,**kwargs):

      self.dtype = kwargs.get('dtype',torch.float64)
      self.device = kwargs.get('device','cpu')

      edges = pd.read_csv(kwargs.get('edges_file','edges.csv'))
      nodes = pd.read_csv(kwargs.get('nodes_file','nodes.csv'))
      self.data = pd.read_csv(kwargs.get('data_file','data.csv'))
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
          sensors.append(sensor)
          m += 1
          next.remove(sensor)
          if m < max_sensors:
            for neighbor in self.G.neighbors(sensor):
              next.append(neighbor)
          else:
            break

      return self.get_fewsensors_dataset(sensors, train = train, dtype = dtype, **kwargs), sensors

    def plot_embeddings(self, limit=5000):
      pos_latlon = nx.circular_layout(self.G)
      pos_graph = nx.circular_layout(self.G)
      lat_max, lat_min = -np.inf, np.inf
      lon_max, lon_min = -np.inf, np.inf

      graph1_max, graph1_min = -np.inf, np.inf
      graph2_max, graph2_min = -np.inf, np.inf

      for node in self.G.nodes():
          emb = self.node_embeddings[node]
          pos_graph[node] = emb[0:2]
          pos_latlon[node] = emb[2:4]

          lat_max, lat_min = max(lat_max, pos_latlon[node][0]), min(lat_min, pos_latlon[node][0])
          lon_max, lon_min = max(lon_max, pos_latlon[node][1]), min(lon_min, pos_latlon[node][1])

          graph1_max, graph1_min = max(graph1_max, pos_graph[node][0]), min(graph1_min, pos_graph[node][0])
          graph2_max, graph2_min = max(graph2_max, pos_graph[node][1]), min(graph2_min, pos_graph[node][1])

      fig, ax = plt.subplots(1, 3, figsize=(15,5))
      nx.draw(self.G, pos_latlon, node_size=25, ax=ax[0], hide_ticks=False)
      ax[0].set_xlim([lat_min, lat_max])
      ax[0].set_ylim([lon_min, lon_max])
      xticks = [k for k in np.linspace(lat_min, lat_max, 5)]
      ax[0].set_xticks(xticks, [str(k) for k in xticks])
      yticks = [k for k in np.linspace(lon_min, lon_max, 5)]
      ax[0].set_yticks(yticks, [str(k) for k in yticks])
      ax[0].tick_params(labelleft=True)

      nx.draw(self.G, pos_graph, node_size=25, ax=ax[1], hide_ticks=False)
      ax[1].set_xlim([graph1_min, graph1_max])
      ax[1].set_ylim([graph2_min, graph2_max])

      ax[2].plot(self.time_embeddings[:limit, 0], color='red', label='Weekly seasonality')
      ax[2].plot(self.time_embeddings[:limit, 1], color='blue', label='Hourly seasonality')
      #ax[2].legend(loc='upper right')
      #limits = plt.axis("on")  # turn off axis
      plt.show()
      #plt.show()



class SensorDataset(Dataset):
  def __init__(self, name, X, y, train = 0.7,
               dtype = torch.float64, **kwargs):
    super().__init__()

    self.NULL_SYMBOL = -99

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
      r = np.random.rand()
      if r >= .7:
        x[0,self.value_index] = torch.tensor([self.NULL_SYMBOL], dtype = self.dtype, device=self.device)
        #x[1:,:] = torch.full((self.max_length-1, self.token_dim),self.NULL_SYMBOL, dtype = self.dtype)
      elif r >= .5:
        x[1:,:] = torch.full((self.max_length-1, self.token_dim),self.NULL_SYMBOL, dtype = self.dtype, device=self.device)
      elif r >= .35:
        x[:,self.value_index] = x[:,self.value_index] + torch.randn(self.max_length, dtype = self.dtype, device=self.device)/12

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