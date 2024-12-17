import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import torch

from st_nca.embeddings.temporal import TemporalEmbedding, to_pandas_datetime
from st_nca.embeddings.spatial import SpatialEmbedding
from st_nca.embeddings.normalization import ZTransform
from st_nca.tokenizer import NeighborhoodTokenizer

from st_nca.common import TensorDictDataframe

from st_nca.datasets.datasets import SensorDataset, AllSensorDataset


class PEMSBase:

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

      self.latlon = kwargs.get("latlon",True)

      if self.latlon:

        coordinates = {}

        for ix, node in enumerate(self.G.nodes()):

            _, lat, lon = nodes[nodes['sensor'] == node].values[0]

            coordinates[node] = {'lat': lat, 'lon': lon }

        nx.set_node_attributes(self.G, coordinates)

      self.node_embeddings = SpatialEmbedding(self.G, latlon=self.latlon, dtype=self.dtype, device=self.device)

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
      
      self.NULL_SYMBOL = self.tokenizer.NULL_SYMBOL

      self.td = kwargs.get('use_tensordict', False)

      if self.td:
        self.to_tensordict()
        

    def to_tensordict(self):
      if not self.td:
        cols1 = self.data.columns[0]
        cols2 = self.data.columns[1:].tolist()

        df1 = self.data[[cols1]]
        df2 = self.data[cols2]

        self.data = TensorDictDataframe(dtype=self.dtype, device = self.device, 
                                        numeric_df=df2, nonnumeric_df=df1)
        self.td = True

    
    def get_sample(self, sensor, index):
      X = self.tokenizer.tokenize_sample(self.data, sensor, index)
      if not self.td:    
        y = torch.tensor(self.data[str(sensor)].values[index+1], dtype=self.dtype, device=self.device)
      else:
        y = self.data[str(sensor),index+1]
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
      if not self.td: 
        return int(self.data.columns[index + 1])
      else:
        return int(self.data.numeric_columns[index])

    
    def to(self, *args, **kwargs):
      if isinstance(args[0], str):
        self.device = args[0]
      else:
        self.dtype = args[0]
      return self


class PEMS03(PEMSBase):
    def __init__(self,**kwargs):
      super(PEMS03, self).__init__(latlon = True, **kwargs)

class PEMS04(PEMSBase):
    def __init__(self,**kwargs):
      super(PEMS03, self).__init__(latlon = False, **kwargs)

class PEMS08(PEMSBase):
    def __init__(self,**kwargs):
      super(PEMS03, self).__init__(latlon = False, **kwargs)