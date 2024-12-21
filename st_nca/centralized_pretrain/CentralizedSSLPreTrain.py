from flautim.pytorch.centralized.Experiment import Experiment
from flautim.pytorch.common import metrics

import torch
from torch import nn

import numpy as np

#from torchmetrics.regression import SymmetricMeanAbsolutePercentageError
from st_nca.common import SMAPE


class CentralizedExperiment(Experiment):
    def __init__(self, model, dataset, measures, logger, context, **kwargs):
        super(CentralizedExperiment, self).__init__(model, dataset, measures, logger, context, **kwargs)

        self.device = kwargs.get('device','cpu')

        self.loss = nn.MSELoss() 
        self.mape = SMAPE
        self.optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
        self.epochs = kwargs.get('epochs', 50)
        self.model = model

    def fit(self, **kwargs):

        self.model = self.model.to(self.device)
        self.logger.log(f"Model training started",  object="experiment_fit", object_id=self.id )

        mse, mape = self.training_loop(self.dataset.dataloader())

        self.logger.log(f"Model training finished",object="experiment_fit", object_id=self.id )

        self.model.save()

        return self.model.get_parameters(), len(self.dataset.dataloader()), {"smape": float(mape), "mse": float(mse)}

    def training_loop(self, data_loader):

        for epoch in range(self.epochs):
            self.logger.log(f"Epoch {epoch}", object="experiment_fit", object_id=self.id )
            errors = []
            mapes = []
            ct = 0
            self.model.train()
            for X,y in data_loader:
                if ct % 500 == 0:
                    self.logger.log(f"Epoch {epoch} Training batch {ct}", object="experiment_evaluate", object_id=self.id )
                ct += 1

                self.optim.zero_grad()

                X = X.to(self.device)
                y = y.to(self.device)

                y_pred = self.model.forward(X)

                error = self.loss(y, y_pred.squeeze())
                map = self.mape(y, y_pred.squeeze())

                error.backward()
                self.optim.step()

                # Grava as métricas de avaliação
                errors.append( error.cpu().item() )
                mapes.append(  map.cpu().item() )
            
            self.measures.log(self, metrics.SMAPE, np.mean(mapes), validation=False, epoch = epoch)
            self.measures.log(self, metrics.MSE, np.mean(errors), validation=False, epoch = epoch)

            self.model.eval()
            errors = []
            mapes = []
            with torch.no_grad():
                ct = 0
                for X,y in data_loader:
                    if ct % 500 == 0:
                        self.logger.log(f"Epoch {epoch} Evaluation batch {ct}", object="experiment_evaluate", object_id=self.id )
                    ct += 1
                    
                    X = X.to(self.device)
                    y = y.to(self.device)
                    y_pred = self.model.forward(X)

                    error= self.loss(y, y_pred.squeeze()).detach().cpu().item()
                    map = self.mape(y, y_pred.squeeze()).detach().cpu().item()

                    errors.append(error)
                    mapes.append(map)

            self.measures.log(self, metrics.SMAPE, np.mean(mapes), validation=True, epoch = epoch)
            self.measures.log(self, metrics.MSE, np.mean(errors), validation=True, epoch = epoch)

            self.model.save()

        return np.mean(errors), np.mean(mapes)
