import torch
from torch import nn

from normalization import ZTransform
from scaling import ScalingTransform

class ValueEmbedding(nn.Module):
  def __init__(self, data, **kwargs):
    super().__init__()
    self.device = kwargs.get('device','cpu')
    self.dtype = kwargs.get('dtype',torch.float32)
    self.type = kwargs.get('value_embedding_type','normalization')

    if self.type == 'normalization':
      self.embedder = ZTransform(data, **kwargs)
    elif self.type == 'scaling':
      self.embedder = ScalingTransform(data, **kwargs)
    else:
      raise ValueError("Unknown embedder type!")
    
  def forward(self, x):
     return self.embedder.forward(x)
  
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.embedder = self.embedder.to(*args, **kwargs)

    return self