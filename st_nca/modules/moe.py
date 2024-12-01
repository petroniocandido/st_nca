import torch
from torch import nn

class Expert(nn.Module):
    def __init__(self, **kwargs):
        super(Expert, self).__init__()
        self.device = kwargs.get('device',2)
        self.dtype = kwargs.get('dtype',2)
        num_layers = kwargs.get('num_layers',2)
        input_dim = kwargs.get('input_dim',2)
        hidden_dim = kwargs.get('hidden_dim',2)
        output_dim = kwargs.get('output_dim',2)

        dropout_pct = kwargs.get('dropout',0.15)

        self.dropout = nn.Dropout(dropout_pct)

        self.normalization = kwargs.get('layernorm',None)

        self.activation = kwargs.get('activation',nn.GELU())

        if num_layers == 1:
            self.layers = nn.ModuleList( [nn.Linear(input_dim, output_dim, device=self.device, dtype=self.dtype)])
        else:
            self.layers = nn.ModuleList( [nn.Linear(input_dim, hidden_dim, device=self.device, dtype=self.dtype)])
            for k in range(1,self.num_layers-1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim, device=self.device, dtype=self.dtype))
            self.layers.append(nn.Linear(hidden_dim, output_dim, device=self.device, dtype=self.dtype))


    def forward(self, x):
        for layer in self.layers:
            if self.normalization is not None:
                x = self.normalization(x)
            x = self.activation(layer(self.dropout(x)))
        return x

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        if isinstance(args[0], str):
            self.device = args[0]
        else:
            self.dtype = args[0]
        self.layers = self.layers.to(*args, **kwargs)
        return self

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.layers = self.layers.train(*args, **kwargs)
        return self


class Gating(nn.Module):
    def __init__(self, **kwargs):
        super(Gating, self).__init__()
        self.device = kwargs.get('device',2)
        self.dtype = kwargs.get('dtype',2)

        input_dim = kwargs.get('input_dim',2)
        hidden_dim = kwargs.get('hidden_dim',2)
        num_experts = kwargs.get('num_experts',2)

        self.dropout = nn.Dropout(0.15)

        self.activation = kwargs.get('activation',nn.GELU())

        # Layers
        self.layer1 = nn.Linear(input_dim, hidden_dim, device=self.device, dtype=self.dtype)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim, device=self.device, dtype=self.dtype)
        self.layer3 = nn.Linear(hidden_dim, num_experts, device=self.device, dtype=self.dtype)


    def forward(self, x):
        x = self.activation(self.layer1(self.dropout(x)))
        x = self.activation(self.layer2(self.dropout(x)))
        x = self.activation(self.layer3(self.dropout(x)))

        return torch.softmax(x, dim=1)
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        if isinstance(args[0], str):
            self.device = args[0]
        else:
            self.dtype = args[0]
        self.layer1 = self.layer1.to(*args, **kwargs)
        self.layer2 = self.layer2.to(*args, **kwargs)
        self.layer3 = self.layer3.to(*args, **kwargs)
        return self

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.layer1 = self.layer1.train(*args, **kwargs)
        self.layer2 = self.layer2.train(*args, **kwargs)
        self.layer3 = self.layer3.train(*args, **kwargs)
        return self
    

class SparseMixtureOfExperts(nn.Module):
    def __init__(self, **kwargs):
        super(SparseMixtureOfExperts, self).__init__()
        self.device = kwargs.get('device',2)
        self.dtype = kwargs.get('dtype',2)
        self.num_experts = kwargs.get('num_experts',4)
        self.activate = kwargs.get('activate',1)

        self.input_dim = kwargs.get('input_dim',1)
        self.output_dim = kwargs.get('output_dim',1)

        gate_hidden_dim = kwargs.get('gate_hidden_dim',1)

        expert_hidden_dim = kwargs.get('expert_hidden_dim',1)

        self.gating = Gating(hidden_dim = gate_hidden_dim,**kwargs)
        
        self.experts = nn.ModuleList([
            Expert(hidden_dim = expert_hidden_dim, **kwargs)
            for k in range(self.num_experts)
        ])
        

    def forward(self, x):
        # Get the weights from the gating network
        weights = self.gating(x)

        indexes = torch.topk(weights,k = self.activate, dim=1)

        # Calculate the expert outputs
        outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=2)

        # Adjust the weights tensor shape to match the expert outputs
        weights = weights.unsqueeze(1).expand_as(outputs)

        # Multiply the expert outputs with the weights and
        # sum along the third dimension
        return torch.sum(outputs * weights, dim=2)