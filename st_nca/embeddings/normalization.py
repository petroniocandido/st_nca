import numpy as np
import pandas as pd
import torch
from torch import nn


class ZTransform(nn.Module):
  def __init__(self, data, **kwargs):
    super().__init__()
    self.device = kwargs.get('device','cpu')
    self.dtype = kwargs.get('dtype',torch.float32)
    data = torch.tensor(data, dtype=self.dtype, device=self.device)
    self.mu = torch.mean(data)
    self.sigma = torch.std(data)
    
  def forward(self, x):
     return (x - self.mu) / self.sigma
  
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.mu = self.mu.to(*args, **kwargs)
    self.sigma = self.sigma.to(*args, **kwargs)

    return self