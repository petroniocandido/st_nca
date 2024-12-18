from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import networkx as nx

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from tensordict import TensorDict

from st_nca.embeddings.temporal import str_to_datetime
from st_nca.cellmodel import CellModel
from st_nca.tokenizer import NeighborhoodTokenizer


def get_timestamp(start_ts, increment_type, increment):
  days = increment if increment_type == 'days' else 0
  seconds = increment if increment_type == 'seconds' else 0
  microseconds = increment if increment_type == 'microseconds' else 0
  milliseconds = increment if increment_type == 'milliseconds' else 0
  minutes = increment if increment_type == 'minutes' else 0
  hours = increment if increment_type == 'hours' else 0
  weeks = increment if increment_type == 'weeks' else 0
  delta = timedelta(days=days, seconds=seconds, microseconds=microseconds, 
                    milliseconds=milliseconds, minutes=minutes, hours=hours, weeks=weeks) 
  return start_ts + delta


def timestamp_generator(start_ts, iterations, increment_type='minutes', step=1):
  for i in range(0, iterations, step):
    yield get_timestamp(start_ts, increment_type, i)


class GraphCellularAutomata(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()

    self.device = kwargs.get('device','cpu')
    self.dtype = kwargs.get('dtype',torch.float32)

    self.graph : nx.Graph = kwargs.get('graph',None)

    self.nodes : list = sorted([k for k in self.graph.nodes()])

    self.num_nodes = len(self.nodes)
    self.max_length = kwargs.get('max_length',None)
    self.token_size = kwargs.get('token_size',None)

    self.tokenizer : NeighborhoodTokenizer  = kwargs.get('tokenizer',None)

    self.tokenizer.graph = self.graph

    self.cell_model : nn.Module = kwargs.get('cell_model',None)

  def build_state(self, date, states) -> TensorDict:
    state = {'timestamp': date}
    for ix, node in enumerate(self.nodes):
      state[str(node)] = states[ix]
    return TensorDict(state)

  # For fine-tunning
  def forward(self, sequences, **kwargs):
    return self.cell_model.forward(sequences)
  
  def step(self, timestamp, current_state):
    tokens = torch.empty(self.num_nodes, self.max_length, self.token_size, dtype=self.dtype, device=self.device)
    for ix, node in enumerate(self.nodes):
      tokens[ix, :, :] = self.tokenizer.tokenize(timestamp, current_state, node)
    return self.forward(tokens) 
  
  def batch_run(self, initial_states, iterations, increment_type='minute', increment=1, **kwargs) -> torch.Tensor:
    batch = len(initial_states['timestamp'])
    state_history = torch.zeros(batch, self.num_nodes * iterations, dtype=self.dtype, device=self.device)
    for ix in range(batch):
      initial_state = TensorDict({key : initial_states[key][ix] for key in initial_states.keys() })
      initial_date = str_to_datetime(initial_state['timestamp'])
      state_history[ix, :] = self.run(initial_date, initial_state, iterations, 
                                      increment_type, increment, return_type='tensor', 
                                      **kwargs)
    return state_history
  
  def run(self, initial_date, initial_state, iterations, increment_type='minute', increment=1, **kwargs) -> torch.Tensor:
    return_type = kwargs.get('return_type','tensordict')
    current_state = TensorDict(initial_state)
    if return_type == 'tensordict':
      state_history = []
    else:
      state_history = torch.zeros(self.num_nodes * iterations, dtype=self.dtype, device=self.device)
    for ix, ts in enumerate(timestamp_generator(initial_date, iterations, increment_type, increment), start=1):
      result = self.step(ts, current_state)
      new_state = self.build_state(ts, result) 
      current_state = new_state

      if return_type == 'tensordict':
        state_history.append(new_state)
      else:
        state_history[self.num_nodes * (ix-1) : self.num_nodes * ix] = result.flatten()

    return state_history
  
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.tokenizer = self.tokenizer.to(*args, **kwargs)
    self.cell_model = self.cell_model.to(*args, **kwargs)
    return self

  def train(self, *args, **kwargs):
    super().train(*args, **kwargs)
    self.cell_model = self.cell_model.train(*args, **kwargs)
    return self
  
  def parameters(self, recurse: bool = True):
    return self.cell_model.parameters(recurse)

    