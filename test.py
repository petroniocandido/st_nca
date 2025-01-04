import torch
from torch import nn

from st_nca.common import resume, get_device, checkpoint_all
from st_nca.datasets.PEMS import PEMS04, get_config as pems_get_config
from st_nca.cellmodel import CellModel, load_config, get_config
from st_nca.pretrain import training_loop

DEVICE = get_device()
DTYPE = torch.float32
#DEFAULT_PATH = 'C:\\Users\\petro\\Dropbox\\Projetos\\futurelab\\posdoc\\st_nca\\st_nca\\st_nca\\'
DEFAULT_PATH = 'D:\\Dropbox\\Projetos\\futurelab\\posdoc\\st_nca\\st_nca\\st_nca\\'
DATA_PATH = DEFAULT_PATH + 'data\\PEMS04\\'
MODELS_PATH = DEFAULT_PATH + 'weights\\PEMS04\\'

STEPS_AHEAD = 1
ITERATIONS = 1

pems = PEMS04(edges_file = DATA_PATH + 'edges.csv', data_file = DATA_PATH + 'data.csv',
    device = DEVICE, dtype = DTYPE, steps_ahead = STEPS_AHEAD)

ds = pems.get_sensor_dataset(300, dtype=torch.float32, behavior='deterministic')