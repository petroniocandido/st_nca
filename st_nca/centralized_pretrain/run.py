from flautim.pytorch.common import run_centralized, get_argparser

import os


import torch
from torch import nn


from st_nca.datasets.PEMS import PEMS03
from st_nca.datasets.datasets import SensorDataset
from st_nca.cellmodel import CellModel as BaseCellModel
from st_nca.common import get_device


import CellModel as FlautimCellModel, CentralizedSSLPreTrain, PEMSDataset as PEMSDataset


DEVICE = get_device()
print(DEVICE)
DTYPE = torch.float32
NTRANSF = 3
NHEADS = 16
NTRANSFF = 1024
TRANSFACT = nn.GELU()
MLP = 3
MLPD = 1024
MLPACT = nn.GELU()
BATCH = 512
EPOCHS = 30

def create_model(pems):
    return BaseCellModel(num_tokens = pems.max_length, dim_token = pems.token_dim,
               num_transformers = NTRANSF, num_heads = NHEADS, feed_forward = NTRANSFF, 
               transformer_activation = TRANSFACT,
               mlp = MLP, mlp_dim = MLPD, mlp_activation = MLPACT,
               dtype = DTYPE, device=DEVICE)


if __name__ == '__main__':


    parser, context, backend, logger, measures = get_argparser()

    pems = PEMS03(steps_ahead=12)
    
    model = FlautimCellModel.FlautimCellModel(context, suffix = 'FL-Global', 
                                        model = create_model(pems))
        
    dataset = PEMSDataset.PEMSDataset(pems = pems, client = 0, batch_size=BATCH, 
                                    type = 'centralized',
                                    xtype = torch.float32, ytype = torch.float32)
        
    experiment = CentralizedSSLPreTrain.CentralizedExperiment(model, dataset, measures, logger, context,
                                            device = DEVICE, epochs = EPOCHS)

    run_centralized(experiment)
