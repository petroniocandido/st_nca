import torch
from torch import nn

from st_nca.modules.transformers import Transformer, get_config as transformer_get_config

from st_nca.modules.moe import SparseMixtureOfExperts
from st_nca.common import activations, dtypes




def load_config(config):
  return CellModel(num_tokens = config.pop('num_tokens',10), 
                   dim_token = config.pop('dim_token',10),
                   num_transformers = config.pop('num_transformers',10), 
                   num_heads = config.pop('num_heads',10), 
                   transformer_feed_forward = config.pop('transformer_feed_forward',10), 
                   transformer_activation = config.pop('transformer_activation',10),
                   feed_forward = config.pop('feed_forward',1),
                   feed_forward_dim = config.pop('feed_forward_dim',100), 
                   feed_forward_activation = config.pop('feed_forward_activation',nn.GELU()),
                   device = config.pop('device',None), 
                   dtype = config.pop('dtype',torch.float32), 
                   **config)


def get_config(model, **extra):
  a = transformer_get_config(model.transformers[0])
  b = { 
    'num_tokens': model.num_tokens, 
    'dim_token': model.dim_token,
    'num_transformers': model.num_transformers,
    'feed_forward': model.feed_forward,
    'feed_forward_dim': model.feed_forward_dim, 
    'feed_forward_activation': model.activation,
    'device': model.device, 
    'dtype': model.dtype
  }
  if model.use_moe:
    b['use_moe'] = model.use_moe
    b['num_experts'] = model.moe.num_experts
  a |= b 
  a |= extra
  return a


class CellModel(nn.Module):
  def __init__(self, num_tokens, dim_token,
               num_transformers, num_heads, transformer_feed_forward, transformer_activation = nn.GELU(),
               feed_forward = 1, feed_forward_dim = 100, feed_forward_activation = nn.ReLU(),
               device = None, dtype = torch.float64, **kwargs):
    super().__init__()
    self.num_tokens = num_tokens
    self.num_transformers = num_transformers
    self.dim_token = dim_token
    self.device = device
    self.dtype = dtype
    self.feed_forward = feed_forward
    self.feed_forward_dim = feed_forward_dim

    self.use_moe = kwargs.get('use_moe',False)
    

    self.transformers = nn.ModuleList([Transformer(num_heads, self.num_tokens, dim_token, transformer_feed_forward, transformer_activation,
                         dtype=self.dtype, device=self.device, **kwargs)
                         for k in range(num_transformers)])

    self.flat = nn.Flatten(1)

    if not self.use_moe:

      self.linear = nn.ModuleList()
      for l in range(feed_forward):
        in_dim = self.num_tokens * self.dim_token if l == 0 else feed_forward_dim
        out_dim = 1 if l == feed_forward-1 else feed_forward_dim
        self.linear.append(nn.Linear(in_dim, out_dim, dtype=self.dtype, device=self.device))

    else:

      num_experts = kwargs.get('num_experts',4)

      self.moe = SparseMixtureOfExperts(dtype=self.dtype, device=self.device,
                                        num_experts = num_experts, activate = 1, 
                                        input_dim = self.num_tokens * self.dim_token,
                                        router_input_dim = self.dim_token,
                                        output_dim = feed_forward_dim,
                                        expert_hidden_dim = feed_forward_dim,
                                        num_layers = self.feed_forward,
                                        activation = feed_forward_activation,
                                        use_moe = False,
                                        #router_hidden_dim = 8,
                                        router_type='lsh')
      self.final = nn.Linear(feed_forward_dim, 1, dtype=self.dtype, device=self.device)

    self.activation = feed_forward_activation
    self.drop = nn.Dropout(.15)


  def forward(self, x):
    for transformer in self.transformers:
      x = transformer(x)
    z = self.flat(x)
    if not self.use_moe:
      for linear in self.linear:
        z = self.activation(linear(self.drop(z)))
    else:
      z = self.moe(z)
      z = self.activation(self.final(self.drop(z)))
    return z

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    for k in range(self.num_transformers):
      self.transformers[k] = self.transformers[k].to(*args, **kwargs)
    if not self.use_moe:
      for k in range(self.mlps):
        self.linear[k] = self.linear[k].to(*args, **kwargs)
    else:
      self.moe = self.moe.to(*args, **kwargs)
      self.final = self.final.to(*args, **kwargs)
    return self

  def train(self, *args, **kwargs):
    super().train(*args, **kwargs)
    for k in range(self.num_transformers):
      self.transformers[k] = self.transformers[k].train(*args, **kwargs)
    if not self.use_moe:
      for k in range(self.mlps):
        self.linear[k] = self.linear[k].train(*args, **kwargs)
    else:
      self.moe = self.moe.train(*args, **kwargs)
      self.final = self.final.train(*args, **kwargs)
    return self