import time
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
#from torchmetrics.regression import SymmetricMeanAbsolutePercentageError

from st_nca.common import SMAPE, checkpoint


def train_step(DEVICE, train, test, model, loss, mape, optim):

  model.train()

  errors = []
  mapes = []

  for X,y in train:

    optim.zero_grad()

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

  batch_size = kwargs.get('batch', 10)

  fig, ax = plt.subplots(1,3, figsize=(15, 5))

  epochs = kwargs.get('epochs', 10)
  lr = kwargs.get('lr', 0.001)
  optimizer = kwargs.get('optim', optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005))

  train_ldr = DataLoader(dataset.train(), batch_size=batch_size, shuffle=True)
  test_ldr = DataLoader(dataset.test(), batch_size=batch_size, shuffle=True)

  loss = nn.MSELoss()
  #mape = SymmetricMeanAbsolutePercentageError().to(DEVICE)
  mape = SMAPE

  error_train = []
  mape_train = []
  error_val = []
  mape_val = []

  start_time = time.time()

  for epoch in range(epochs):

    if epoch % 5 == 0:
      checkpoint(model, checkpoint_file)

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