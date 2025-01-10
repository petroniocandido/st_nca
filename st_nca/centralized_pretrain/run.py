from flautim.pytorch.common import run_centralized, get_argparser

import os


import torch
from torch import nn


from st_nca.datasets.PEMS import PEMS03, PEMS04, PEMS08
from st_nca.datasets.datasets import SensorDataset
from st_nca.cellmodel import CellModel as BaseCellModel, load_config
from st_nca.common import get_device


import CellModel as FlautimCellModel, CentralizedSSLPreTrain, PEMSDataset as PEMSDataset


DEVICE = get_device()
DTYPE = torch.float32
H = 12
BATCH = 1024
EPOCHS = 50
LR = 0.001

print(DEVICE)

def create_model(pems):
    return load_config( {
        'num_tokens': pems.max_length,
        'dim_token': pems.token_dim,
        'num_transformers': 3,
        'num_heads': 16,
        'transformer_feed_forward': 1024,
        'transformer_activation': nn.GELU(approximate='none'),
        'normalization': torch.nn.modules.normalization.LayerNorm,
        'pre_norm': False,
        'feed_forward': 3,
        'feed_forward_dim': 1024,
        'feed_forward_activation': nn.GELU(approximate='none'),
        'device': DEVICE,
        'dtype': DTYPE,
        'steps_ahead': H
        }
)

if __name__ == '__main__':


    parser, context, backend, logger, measures = get_argparser()

    pems = PEMS04(steps_ahead=H)
    
    model = FlautimCellModel.FlautimCellModel(context, suffix = 'FL-Global', 
                                        model = create_model(pems))
        
    dataset = PEMSDataset.PEMSDataset(pems = pems, client = 0, batch_size=BATCH, 
                                    type = 'centralized',
                                    xtype = DTYPE, ytype = DTYPE)
        
    experiment = CentralizedSSLPreTrain.CentralizedExperiment(model, dataset, measures, logger, context,
                                            device = DEVICE, epochs = EPOCHS,
                                            lr = LR)

    run_centralized(experiment)
