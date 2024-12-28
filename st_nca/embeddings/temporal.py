from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from torch import nn
from tensordict import TensorDict


def datetime_to_str(dt):
  return dt.strftime("%Y%m%d%H%M%S")

def str_to_datetime(dt):
  return datetime.strptime(dt, '%Y%m%d%H%M%S')

def to_pandas_datetime(values):
  return pd.to_datetime(values, format='%m/%d/%Y %H:%M')

def from_np_to_datetime(dt):
  dt = to_pandas_datetime(dt)
  return datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)

def from_pd_to_datetime(dt):
  return datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)
  
def from_datetime_to_pd(date : datetime):
  #return to_pandas_datetime(np.datetime64(date.astimezone(timezone.utc)))
  return to_pandas_datetime(np.datetime64(date.astimezone(None)))


class TemporalEmbedding(nn.Module):
  def __init__(self, dates, **kwargs):
    super().__init__()
    self.device = kwargs.get('device','cpu')
    self.dtype = kwargs.get('dtype',torch.float32)
    self.pi2 = torch.tensor([2 * torch.pi], dtype=self.dtype, device=self.device)
    self.week_minutes_rads = torch.tensor([(7 * 1440) * self.pi2], dtype=self.dtype, device=self.device)
    tmp_dict = {}
    self.length = 0
    for date in dates:
      tmp_dict[datetime_to_str(date)] = self.forward(date)
      self.length += 1 
    self.embeddings : TensorDict = TensorDict(tmp_dict) 

  def week_embedding(self, date):
      day_of_week = date.isocalendar()[2]  # Day of week (1: Monday, 7: Sunday)
      num = torch.tensor([day_of_week * 1440 + date.hour * 60 + date.minute], dtype=self.dtype, device=self.device)
      return torch.sin(num / self.week_minutes_rads)

  def minute_embedding(self, date):
    minute_of_day = torch.tensor([(date.hour * 60 + date.minute)/ 1440], dtype=self.dtype, device=self.device)
    return torch.sin(minute_of_day * self.pi2)
  
  def forward(self, dt):
     date = from_pd_to_datetime(dt)
     we = self.week_embedding(date)
     me = self.minute_embedding(date)
     return torch.tensor([we, me], dtype=self.dtype, device=self.device)
  
  def __getitem__(self, date):
    if isinstance(date, np.datetime64):
      date =  from_np_to_datetime(date)
      return self.embeddings[datetime_to_str(date)]
    elif isinstance(date, datetime):
      return self.embeddings[datetime_to_str(date)]
    elif isinstance(date, int):
      return self.embeddings[date]
    else:
      raise Exception("Unknown index type")
    
  
  def all(self):
    ret = torch.empty(self.length, 2,
                        dtype=self.dtype, device=self.device)
    for it,emb in enumerate(self.embeddings.values(sort=True)):
      ret[it, :] = emb
    return ret
  
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.pi2 = self.pi2.to(*args, **kwargs)
    self.week_minutes_rads = self.week_minutes_rads.to(*args, **kwargs)
    return self