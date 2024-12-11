import torch
from torch import nn

from st_nca.modules.transformers import Transformer

from st_nca.modules.moe import SparseMixtureOfExperts

class CellModel(nn.Module):
  def __init__(self, num_tokens, dim_token,
               num_transformers, num_heads, feed_forward, transformer_activation = nn.GELU(),
               mlp = 1, mlp_dim = 100, mlp_activation = nn.ReLU(),
               device = None, dtype = torch.float64, **kwargs):
    super().__init__()
    self.num_tokens = num_tokens
    self.num_transformers = num_transformers
    self.dim_token = dim_token
    self.device = device
    self.dtype = dtype
    self.mlps = mlp

    self.use_moe = kwargs.get('use_moe',False)
    

    self.transformers = nn.ModuleList([Transformer(num_heads, self.num_tokens, dim_token, feed_forward, transformer_activation,
                         dtype=self.dtype, device=self.device)
                         for k in range(num_transformers)])

    self.flat = nn.Flatten(1)

    if not self.use_moe:

      self.linear = nn.ModuleList()
      for l in range(mlp):
        in_dim = self.num_tokens * self.dim_token if l == 0 else mlp_dim
        out_dim = 1 if l == mlp-1 else mlp_dim
        self.linear.append(nn.Linear(in_dim, out_dim, dtype=self.dtype, device=self.device))

    else:

      num_experts = kwargs.get('num_experts',4)

      self.moe = SparseMixtureOfExperts(dtype=self.dtype, device=self.device,
                                        num_experts = num_experts, activate = 1, 
                                        input_dim = self.num_tokens * self.dim_token,
                                        router_input_dim = self.dim_token,
                                        output_dim = mlp_dim,
                                        expert_hidden_dim = mlp_dim,
                                        num_layers = self.mlps,
                                        activation = mlp_activation,
                                        use_moe = False,
                                        #router_hidden_dim = 8,
                                        router_type='lsh')
      self.final = nn.Linear(mlp_dim, 1, dtype=self.dtype, device=self.device)

    self.activation = mlp_activation
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