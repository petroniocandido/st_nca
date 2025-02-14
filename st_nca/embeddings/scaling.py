import torch
from torch import nn


class ScalingTransform(nn.Module):
  def __init__(self, data, **kwargs):
    super().__init__()
    self.device = kwargs.get('device','cpu')
    self.dtype = kwargs.get('dtype',torch.float32)
    if not isinstance(data, torch.Tensor):
      data = torch.tensor(data, dtype=self.dtype, device=self.device)
    self.min = torch.min(torch.nan_to_num(data,0,0,0))
    max = torch.max(torch.nan_to_num(data,0,0,0))
    self.range = max - self.min
    
  def forward(self, x):
     return (x - self.min) / self.range
  
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.min = self.min.to(*args, **kwargs)
    self.range = self.range.to(*args, **kwargs)

    return self