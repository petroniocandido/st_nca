from flautim.pytorch.federated.Experiment import Experiment
from flautim.pytorch.common import metrics
import MNISTModel, MNISTDataset
import numpy as np
import time
import torch

class MNISTExperiment(Experiment):
    def __init__(self, model, dataset, measures, logger, context, **kwargs):
        super(MNISTExperiment, self).__init__(model, dataset, measures, logger, context, **kwargs)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.epochs = kwargs.get('epochs', 1)

    def training_loop(self, data_loader):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model.train()
        correct = 0
        custo = 0.0
        for epoca in range(self.epochs):
            
            for images, labels in data_loader:
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                custo += np.mean(loss.cpu().item())
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
            
        return custo, correct/len(self.dataset)

    def validation_loop(self, data_loader):
        loss = 0.0
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for images, labels in data_loader:
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()/ len(labels)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        accuracy = correct / len(self.dataset)

        return loss, accuracy
        
    def weighted_average(self, FL_metrics, server_round):
    
        accuracies = [ m["accuracy"] for num_examples, m in FL_metrics]
        losses = [ m["loss"] for num_examples, m in FL_metrics]
    
        examples = [num_examples for num_examples, _ in FL_metrics]
        
        self.epoch_fl = server_round
        
        self.measures.log(self, metrics.CROSSENTROPY, sum(accuracies) / len(accuracies), validation=True)
        self.measures.log(self, metrics.ACCURACY, sum(losses) / len(losses), validation=True)

        return {"accuracy": sum(accuracies) / len(accuracies)}