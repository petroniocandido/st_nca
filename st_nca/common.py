
import torch

def MAPE(y, y_pred):
  return torch.mean((y - y_pred).abs() / (y.abs() + 1e-8))

def SMAPE(y, y_pred):
  return torch.mean(2*(y - y_pred).abs() / (y.abs() + y_pred.abs() + 1e-8))

def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'

def checkpoint(model, file):
  torch.save(model.state_dict(), file)

def checkpoint_all(model, optimizer, file):
  torch.save({
    'optim': optimizer.state_dict(),
    'model': model.state_dict(),
}, file)

def resume(model, file):
  model.load_state_dict(torch.load(file, weights_only=True, map_location=torch.device(get_device())))

def resume_all(model, optimizer, file):
  checkpoint = torch.load( file, map_location=torch.device(get_device()))
  model.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optim'])