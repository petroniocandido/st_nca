import numpy as np
import pandas as pd
import torch
from torch import nn

from tensordict import TensorDict

from st_nca.embeddings import temporal, spatial, normalization


class NeighborhoodTokenizer(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()

    self.NULL_SYMBOL = 0

    self.device = kwargs.get('device','cpu')
    self.dtype = kwargs.get('dtype',torch.float32)

    self.graph = kwargs.get('graph',None)

    self.num_nodes = kwargs.get('num_nodes',None)
    self.max_length = kwargs.get('max_length',None)
    self.token_dim = kwargs.get('token_dim',None)

    self.ztransform = kwargs.get('ztransform',None)
    self.spatial_embedding = kwargs.get('spatial_embedding',None)
    self.temporal_embedding = kwargs.get('temporal_embedding',None)

  def normalized_data(self, data, sensor):
    if isinstance(data, pd.DataFrame):
      values = data[str(sensor)].values
    elif isinstance(data, TensorDict):
      values = data[str(sensor)]
                
    return self.ztransform(torch.tensor(values, dtype=self.dtype, device=self.device))
    
  def normalized_sample(self, data, sensor, index):
    if isinstance(data, pd.DataFrame):
      value = data[str(sensor)].values[index]
    elif isinstance(data, TensorDict):
      value = data[str(sensor)]
    return self.ztransform(torch.tensor(value, dtype=self.dtype, device=self.device))
  
  
  def tokenize(self, timestamp, values, node):
    val = self.ztransform(values[str(node)])
    tim_emb = self.temporal_embedding(timestamp)

    tokens = self.spatial_embedding[node]
    tokens = torch.hstack([tokens, val ])
    tokens = torch.hstack([tokens, tim_emb])

    m = 1

    for neighbor in self.graph.neighbors(node):
      m += 1
      tokens = torch.hstack([tokens, self.spatial_embedding[neighbor]])
      tokens = torch.hstack([tokens, self.ztransform(values[str(neighbor)])])
      tokens = torch.hstack([tokens, tim_emb])

    tokens = tokens.reshape(1, m, self.token_dim)

    tokens = torch.hstack([tokens, torch.full((1, self.max_length - m, self.token_dim), self.NULL_SYMBOL)])

    return tokens
  
  # Create an empty sequence of tokens (filled with -1)
  # Then extracts the data for a sensor and its neighbors and fill the values
  # in the tokens, together with the spatial and temporal embeddings
  def tokenize_all(self, data, sensor):

    tmp = self.normalized_data(data, sensor)
    n = len(tmp)
    tim_emb = self.temporal_embedding.all().reshape(n,2)

    tokens = self.spatial_embedding[sensor].repeat(n,1)
    tokens = torch.hstack([tokens, tmp.reshape(n,1) ])
    tokens = torch.hstack([tokens, tim_emb])

    m = 1

    for neighbor in self.graph.neighbors(sensor):
      m += 1
      tokens = torch.hstack([tokens, self.spatial_embedding[neighbor].repeat(n,1)])
      tokens = torch.hstack([tokens, self.normalized_data(data, neighbor).reshape(n,1) ])
      tokens = torch.hstack([tokens, tim_emb])

    tokens = tokens.reshape(n, m, self.token_dim)


    tokens = torch.hstack([tokens, torch.full((n,  self.max_length - m, self.token_dim), self.NULL_SYMBOL)])

    return tokens

  def tokenize_sample(self, data, node, index):

    if isinstance(data, pd.DataFrame):
      dt = data['timestamp'][index]
    elif isinstance(data, TensorDict):
      dt = index

    tim_emb = self.temporal_embedding[dt]

    tokens = self.spatial_embedding[node]
    tokens = torch.hstack([tokens, self.normalized_sample(data, node, index)])
    tokens = torch.hstack([tokens, tim_emb])

    m = 1

    for neighbor in self.graph.neighbors(node):
      m += 1
      tokens = torch.hstack([tokens, self.spatial_embedding[neighbor]])
      tokens = torch.hstack([tokens, self.normalized_sample(data, neighbor, index)])
      tokens = torch.hstack([tokens, tim_emb])

    tokens = tokens.reshape(1, m, self.token_dim)

    tokens = torch.hstack([tokens, torch.full((1, self.max_length - m, self.token_dim), self.NULL_SYMBOL)])

    return tokens.reshape(self.max_length, self.token_dim)
    
  def forward(self, data, node, sample=None, **kwargs):
     if sample is None:
       return self.tokenize_all(data, node)
     else:
       return self.tokenize_sample(data, node, sample)
  
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.ztransform = self.ztransform.to(*args, **kwargs)
    self.spatial_embedding = self.spatial_embedding.to(*args, **kwargs)
    self.temporal_embedding = self.temporal_embedding.to(*args, **kwargs)
    return self