from flautim.pytorch.federated.Experiment import Experiment
from flautim.pytorch.common import metrics

import numpy as np
import time
import torch
from torch import nn

from torchmetrics.regression import SymmetricMeanAbsolutePercentageError

class MNISTExperiment(Experiment):
    def __init__(self, model, dataset, measures, logger, context, **kwargs):
        super(MNISTExperiment, self).__init__(model, dataset, measures, logger, context, **kwargs)

        self.device = kwargs.get('device','cpu')

        self.loss = nn.MSELoss()
        self.mape = SymmetricMeanAbsolutePercentageError()
        self.optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
        self.epochs = kwargs.get('epochs', 1)
        self.model = model

    def fit(self, parameters, config):
        self.logger.log("Model training started", details="", object="experiment_fit", object_id=self.id )

        self.model.set_parameters(parameters)
        
        self.epoch_fl = config["server_round"]

        mse, mape = self.training_loop(self.dataset.dataloader())

        self.logger.log("Model training finished", details="", object="experiment_fit", object_id=self.id )

        self.model.save()

        return self.model.get_parameters(), len(self.dataset.dataloader()), {"smape": float(mape), "mse": float(mse)}

    def evaluate(self, parameters, config):

        self.logger.log("Model evaluation started", details="", object="experiment_evaluate", object_id=self.id )
        
        self.model.set_parameters(parameters)
        
        mse, mape = self.validation_loop(self.dataset.dataloader(validation = True))

        self.logger.log("Model training finished", details="", object="experiment_evaluate", object_id=self.id )
        
        self.model.save()
        
        return float(mse), len(self.dataset.dataloader(validation = True)),  {"smape": float(mape), "mse": float(mse)}

    def training_loop(self, data_loader):
        self.model.train()

        errors = []
        mapes = []
        for X,y in data_loader:

            self.optim.zero_grad()

            X = X.to(self.device)
            y = y.to(self.device)

            y_pred = self.model.forward(X)

            error = self.loss(y, y_pred.squeeze())
            map = self.mape(y, y_pred.squeeze())

            error.backward()
            self.optim.step()

            # Grava as métricas de avaliação
            errors.append(error.cpu().item())
            mapes.append(map.cpu().item())

        return errors, mapes

    def validation_loop(self, data_loader):
        self.model.eval()
        errors = []
        mapes = []
        with torch.no_grad():
            for X,y in data_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                y_pred = self.model.forward(X)

                error= self.loss(y, y_pred.squeeze()).detach().cpu().item()
                map = self.mape(y, y_pred.squeeze()).detach().cpu().item()

                errors.append(error)
                mapes.append(map)

        return errors, mapes
    
    def weighted_average(self, FL_metrics, server_round):
    
        smape = [ m["smape"] for num_examples, m in FL_metrics]
        mse = [ m["loss"] for num_examples, m in FL_metrics]
    
        examples = [num_examples for num_examples, _ in FL_metrics]
        
        self.epoch_fl = server_round
        
        self.measures.log(self, metrics.SMAPE, sum(smape) / len(smape), validation=True)
        self.measures.log(self, metrics.MSE, sum(mse) / len(mse), validation=True)

        return {"smape": sum(smape) / len(smape)}