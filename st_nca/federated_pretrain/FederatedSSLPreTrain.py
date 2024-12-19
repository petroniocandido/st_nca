from flautim.pytorch.federated.Experiment import Experiment
from flautim.pytorch.common import metrics

import torch
import numpy as np
from torch import nn

#from torchmetrics.regression import SymmetricMeanAbsolutePercentageError
from st_nca.common import SMAPE


class FederatedExperiment(Experiment):
    def __init__(self, model, dataset, measures, logger, context, **kwargs):
        super(FederatedExperiment, self).__init__(model, dataset, measures, logger, context, **kwargs)

        self.device = kwargs.get('device','cpu')

        self.loss = nn.MSELoss() 
        self.mape = SMAPE
        self.optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
        self.epochs = kwargs.get('epochs', 50)
        self.model = model
        self.log_msg_infix = f' Client {self.dataset.client} Model {self.dataset.sensor} '

    def fit(self, parameters, config):
        self.epoch_fl = config["server_round"]
        prefix = f"Round {self.epoch_fl} " + self.log_msg_infix
        self.model = self.model.to(self.device)
        self.logger.log(prefix + " training started", object="experiment_fit", object_id=self.id )

        self.model.set_parameters(parameters)

        mse, mape = self.training_loop(self.dataset.dataloader())

        self.logger.log(prefix + " training finished", object="experiment_fit", object_id=self.id )

        self.measures.log(self, metrics.SMAPE, mape, validation=False)
        self.measures.log(self, metrics.MSE, mse, validation=False)

        self.model.save()

        return self.model.get_parameters(), len(self.dataset.dataloader()), {"smape": float(mape), "mse": float(mse)}

    def evaluate(self, parameters, config):

        prefix = f"Round {self.epoch_fl} " + self.log_msg_infix

        self.logger.log(prefix + " evaluation started", object="experiment_evaluate", object_id=self.id )
        
        self.model.set_parameters(parameters)
        
        mse, mape = self.validation_loop(self.dataset.dataloader(validation = True))

        self.logger.log(prefix + " evaluation finished", object="experiment_evaluate", object_id=self.id )
        
        self.model.save()
        
        return float(mse), len(self.dataset.dataloader(validation = True)),  {"smape": float(mape), "mse": float(mse)}

    def training_loop(self, data_loader):
        self.model.train()

        prefix = f"Round {self.epoch_fl} " + self.log_msg_infix

        errors = None
        mapes = None
        for epoch in range(self.epochs):
            self.logger.log(prefix + " Training Epoch {epoch}", object="experiment_fit", object_id=self.id )
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
                errors = error.cpu().item()
                mapes = map.cpu().item()

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

        return np.mean(errors), np.mean(mapes)
    
    def weighted_average(self, FL_metrics, server_round):
    
        smape = [ m["smape"] for num_examples, m in FL_metrics]
        mse = [ m["mse"] for num_examples, m in FL_metrics]
    
        examples = [num_examples for num_examples, _ in FL_metrics]
        
        self.epoch_fl = server_round
        
        self.measures.log(self, metrics.SMAPE, sum(smape) / len(smape), validation=True)
        self.measures.log(self, metrics.MSE, sum(mse) / len(mse), validation=True)

        return {"smape": sum(smape) / len(smape)}
