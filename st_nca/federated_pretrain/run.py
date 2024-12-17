from flautim.pytorch.common import run_federated, get_argparser

import os


import torch
from torch import nn


from st_nca.datasets import PEMS03, SensorDataset
from st_nca.cellmodel import CellModel as BaseCellModel
from st_nca.common import get_device


import CellModel as FlautimCellModel, FederatedSSLPreTrain, PEMS03Dataset


DEVICE = get_device()
print(DEVICE)
DTYPE = torch.float32
NTRANSF = 2
NHEADS = 8
NTRANSFF = 512
TRANSFACT = nn.GELU()
MLP = 2
MLPD = 512
MLPACT = nn.GELU()

def create_model(pems):
    return BaseCellModel(num_tokens = pems.max_length, dim_token = pems.token_dim,
               num_transformers = NTRANSF, num_heads = NHEADS, feed_forward = NTRANSFF, 
               transformer_activation = TRANSFACT,
               mlp = MLP, mlp_dim = MLPD, mlp_activation = MLPACT,
               dtype = DTYPE, device=DEVICE)

def generate_client_fn(pems, context, measures, logger):
    
    def create_client_fn(ctx):

        if str(ctx) != 'FL-Global':
            sensor = pems.get_sensor(int(ctx.node_config["partition-id"]))
            id = int(ctx.node_config["partition-id"])
        else:
            sensor = 'FL-Global'
            id = 0
        
        model = FlautimCellModel.FlautimCellModel(context, suffix = str(sensor), 
                                        model = create_model(pems))
        
        dataset = PEMS03Dataset.PEMS03Dataset(pems = pems, client = id, batch_size=512, 
                                        xtype = torch.float32, ytype = torch.float32)
        
        return FederatedSSLPreTrain.FederatedExperiment(model, dataset, measures, logger, context,
                                               device = DEVICE, epochs = 5)
        
    return create_client_fn
    

def evaluate_fn(pems, context, measures, logger):
    def fn(server_round, parameters, config):
        """This function is executed by the strategy it will instantiate
        a model and replace its parameters with those from the global model.
        The, the model will be evaluate on the test set (recall this is the
        whole MNIST test set)."""

        model = FlautimCellModel.FlautimCellModel(context, model = create_model(pems))
        model.set_parameters(parameters)
        
        dataset = PEMS03Dataset.PEMS03Dataset(batch_size=2048, client = 0, pems = pems,
                                        xtype = torch.float32, ytype = torch.float32)
        
        experiment = FederatedSSLPreTrain.FederatedExperiment(model, dataset, measures, logger, context,
                                               device = DEVICE)
        
        mse, smape = experiment.validation_loop(dataset.dataloader(validation=True)) 

        return mse, {"smape": smape}

    return fn

if __name__ == '__main__':


    parser, context, backend, logger, measures = get_argparser()

    edges_file = "https://raw.githubusercontent.com/petroniocandido/st_nca/refs/heads/main/st_nca/data/PEMS03/edges.csv"
    nodes_file = "https://raw.githubusercontent.com/petroniocandido/st_nca/refs/heads/main/st_nca/data/PEMS03/nodes.csv"
    data_file = "https://raw.githubusercontent.com/petroniocandido/st_nca/refs/heads/main/st_nca/data/PEMS03/data.csv"

    pems = PEMS03(edges_file=edges_file, nodes_file=nodes_file, data_file=data_file)
    
    client_fn_callback = generate_client_fn(pems, context, measures, logger)
    evaluate_fn_callback = evaluate_fn(pems, context, measures, logger)

    run_federated(client_fn_callback, evaluate_fn_callback, 
                  num_clients = pems.num_sensors, num_rounds = 500 )
