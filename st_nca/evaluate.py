import numpy as np
import pandas as pd
import torch

from st_nca.embeddings.temporal import str_to_datetime, from_datetime_to_pd
from st_nca.common import SMAPE, MAE, RMSE, nRMSE


def diff_states(state1, state2):
  keys1 = [k for k in sorted(state1.keys())]
  keys2 = [k for k in sorted(state2.keys())]
  if len(keys1) != len(keys2):
    raise ValueError("Different number of keys")
  acc = []
  for k in keys1:
    v1 = state1[k]
    v2 = state2[k]
    if isinstance(v1, (pd.Timestamp, str)):
      continue
    if isinstance(v1, torch.Tensor):
      v1 = v1.cpu().detach().numpy()[0]
    if isinstance(v2, torch.Tensor):
      v2 = v2.cpu().detach().numpy()[0]
    acc.append( np.abs(v1 - v2) )
  acc = np.array(acc)
  return np.min(acc), np.median(acc), np.mean(acc), np.std(acc), np.max(acc)


def extract_tensor(model, state):
  n = len(model.nodes)
  vals = [state[k] for k in model.nodes]
  return torch.tensor(vals, device=model.device, dtype=model.dtype)


def evaluate(dataset, gca, steps_ahead, increment_type='minutes', increment=5):
  columns = ['timestamp','mape','mae','rmse','nrmse']
  rows = []
  for X,y in dataset:
    p = gca.run(str_to_datetime(X['timestamp']), X, iterations=steps_ahead, 
                increment_type=increment_type, increment=increment, 
                return_type='tensordict')
    row = [y['timestamp']]
    y_ = extract_tensor(gca, y)
    p_ = extract_tensor(gca, p[-1])
    row.extend([SMAPE(y_, p_), MAE(y_, p_), RMSE(y_, p_), nRMSE(y_, p_)])
    rows.append(row)
  return pd.DataFrame(rows, columns=columns)