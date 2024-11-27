
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def checkpoint(model, file):
  torch.save(model.state_dict(), file)

def checkpoint_all(model, optimizer, file):
  torch.save({
    'optim': optimizer.state_dict(),
    'model': model.state_dict(),
}, file)

def resume(model, file):
  model.load_state_dict(torch.load(file, weights_only=True, map_location=torch.device(DEVICE)))

def resume_all(model, optimizer, file):
  checkpoint = torch.load( file, map_location=torch.device(DEVICE))
  model.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optim'])