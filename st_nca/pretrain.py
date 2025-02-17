import time
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
#from torchmetrics.regression import SymmetricMeanAbsolutePercentageError

from st_nca.common import checkpoint
from st_nca.evaluate import SMAPE, MAPE


def train_step(DEVICE, train, test, model, loss, mape, optim):

  model.train()

  errors = []
  mapes = []

  for X,y in train:

    # Batch times
    #start_time = time.time()

    #Performance advice found at: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    #optim.zero_grad()
    for param in model.parameters():
      param.grad = None

    X = X.to(DEVICE)
    y = y.to(DEVICE)

    y_pred = model.forward(X)

    error = loss(y, y_pred.squeeze())
    map = mape(y, y_pred.squeeze())

    error.backward()
    optim.step()

    # Grava as métricas de avaliação
    errors.append(error.cpu().item())
    mapes.append(map.cpu().item())

    #print batch times
    #print(round(time.time() - start_time, 3))


  ##################
  # VALIDATION
  ##################

  model.eval()

  errors_val = []
  mapes_val = []
  with torch.no_grad():
    for X,y in test:

      X = X.to(DEVICE)
      y = y.to(DEVICE)

      y_pred = model(X)

      error_val = loss(y, y_pred.squeeze())
      map_val = mape(y, y_pred.squeeze())

      errors_val.append(error_val.cpu().item())
      mapes_val.append(map_val.cpu().item())


  return errors, mapes, errors_val, mapes_val


def training_loop(DEVICE, dataset, model, display = None, **kwargs):

  model = model.to(DEVICE)

  checkpoint_file = kwargs.get('checkpoint_file', 'modelo.pt')

  if display is None:
    from IPython import display

  

  fig, ax = plt.subplots(1,3, figsize=(15, 5))

  epochs = kwargs.get('epochs', 10)
  lr = kwargs.get('lr', 0.001)
  optimizer = kwargs.get('optim', optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005))

  loss = nn.MSELoss()
  #mape = SymmetricMeanAbsolutePercentageError().to(DEVICE)
  mape = SMAPE

  dynamic_batch = kwargs.get('dynamic_batch', False)

  if dynamic_batch:
    batch_schedule = [2048, 1024, 512, 256]
    bucket = epochs // 4
  else:
    batch_size = kwargs.get('batch', 10)
    train_ldr = DataLoader(dataset.train(), batch_size=batch_size, shuffle=True, num_workers=2)
    test_ldr = DataLoader(dataset.test(), batch_size=batch_size, shuffle=True, num_workers=2)
  

  error_train = []
  mape_train = []
  error_val = []
  mape_val = []

  start_time = time.time()

  for epoch in range(epochs):

    checkpoint(model, checkpoint_file)

    if dynamic_batch:
      batch_size = batch_schedule[epoch // bucket]
      train_ldr = DataLoader(dataset.train(), batch_size=batch_size, shuffle=True, num_workers=2)
      test_ldr = DataLoader(dataset.test(), batch_size=batch_size, shuffle=True, num_workers=2)

    errors_train, map_train, errors_val, map_val = train_step(DEVICE, train_ldr, test_ldr, model, loss, mape, optimizer)

    error_train.append(np.mean(errors_train))
    mape_train.append(np.mean(map_train))
    error_val.append(np.mean(errors_val))
    mape_val.append(np.mean(map_val))

    display.clear_output(wait=True)
    ax[0].clear()
    ax[0].plot(error_train, c='blue', label='Train')
    ax[0].plot(error_val, c='red', label='Test')
    ax[0].legend(loc='upper left')
    ax[0].set_title("LOSS - All Epochs {} - Time: {} s".format(epoch, round(time.time() - start_time, 0)))
    ax[1].clear()
    ax[1].plot(error_train[-20:], c='blue', label='Train')
    ax[1].plot(error_val[-20:], c='red', label='Test')
    ax[1].set_title("LOSS - Last 20 Epochs".format(epoch))
    ax[1].legend(loc='upper left')
    ax[2].clear()
    ax[2].plot(mape_train[-20:], c='blue', label='Train')
    ax[2].plot(mape_val[-20:], c='red', label='Test')
    ax[2].set_title("MAPE - Last 20 Epochs".format(epoch))
    ax[2].legend(loc='upper left')
    plt.tight_layout()
    display.display(plt.gcf())

  plt.savefig(checkpoint_file+".pdf", dpi=150)

  checkpoint(model, checkpoint_file)


def evaluate(model, in_sample, out_sample, num_samples = None):
  metrics = {"MAPE": MAPE, "SMAPE": SMAPE}

  model.eval()

  results = {}

  results['train'] = experiment_on_dataset(model, in_sample, num_samples, metrics)
  results['test'] = experiment_on_dataset(model, out_sample, num_samples, metrics)

  return results

def experiment_on_dataset(model, sample, num_samples, metrics):
    total = len(sample)
    samples = num_samples if not num_samples is None else total
    indexes = torch.randperm(samples, device=model.device)

    X_batch = torch.zeros(samples, model.num_tokens, model.dim_token, 
                      device=model.device, dtype=model.dtype)
    y_batch = torch.zeros(samples, device=model.device, dtype=model.dtype)
  
    for ct,ix in enumerate(indexes):
      X,y = sample[ix]
      X_batch[ct,:] = X
      y_batch[ct] = y

    out = model(X_batch)

    res = {}
    for key, metric in metrics.items():
      res[key] = metric(y_batch, out).detach().cpu()

    return res