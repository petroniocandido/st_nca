from flautim.pytorch.Model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlautimCellModel(Model):
    def __init__(self, context, **kwargs):
        super(FlautimCellModel, self).__init__(context, name = "FLAUTIMCellModel", version = 1, id = 1, **kwargs)

        self.model = kwargs.get('model', None)

    def forward(self, x):
        return self.model(x)
    
    def parameters(self):
        return self.model.parameters()
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.model = self.model.to(*args, **kwargs)
        return self

    def train(self, *args, **kwargs):
        self = super().train(*args, **kwargs)
        self.model = self.model.train(*args, **kwargs)
        return self