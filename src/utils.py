import sklearn.preprocessing as sk

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import sampler
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def getData(data, target, window, train_split, \
   batch_size=20, future=0):
   '''
     Implement custom function for creating 
     datasets for training.
     Parameters: 
        - data: list
        - target: list
        - window: int, the length of x interval
        - train_split: int the number of train samples
        - batch_size: int
        - future: int, default 0,the length of prediction wanted
     Output:
        - result: 
   '''
   x_vals = []
   y_vals = []
   start = 0 + window
   end = len(data) - future

   for i in range(start, end):
      x = data[(i-window):i,:]
      y = target[i:(i+future+1),:]
      x_vals.append(x)
      y_vals.append(y)
        
   tensor_x = torch.Tensor(x_vals)
   tensor_y = torch.Tensor(y_vals)

   set_train = TensorDataset(tensor_x,tensor_y)
   print(set_train.shape)
   loader_train = DataLoader(set_train, batch_size = batch_size,\
      sampler=sampler.SubsetRandomSampler(range(train_split)))
   loader_test = DataLoader(set_train, batch_size = batch_size,\
      sampler=sampler.SubsetRandomSampler(range(train_split, len(set_train))))
   
   return (loader_train,loader_test)

def vwap(df):
   '''
   Compute the vwap.
      Params:
         - df: pandas.dataframe object of all 5 columns
      Out:
         - list: of floats
   '''
   df_close_hour = np.array(df['Close'])
   df_high_hour = np.array(df['High'])
   df_low_hour = np.array(df['Low'])
   df_volume_hour = np.array(df['Volume'])

   residual = (df_close_hour.size) % 24
   vwap = []
   for i in range(df_close_hour.size//24):

      close_hour_sub = df_close_hour[i*24 + residual: (i+1)*24 + residual] #list of 24

      high_hour_sub = df_high_hour[i*24 + residual: (i+1)*24 + residual ] #list of 24

      low_hour_sub = df_low_hour[i*24 + residual : (i+1)*24 + residual ] #list of 24

      volume_hour_sub = df_volume_hour[i*24 + residual : (i+1)*24 + residual ] # 24hr trade volumes list of 24

      average_hour_sub = ((close_hour_sub + high_hour_sub + low_hour_sub)/3) * volume_hour_sub #list of 24
    
      if sum(volume_hour_sub) == 0:
         vwap_day = close_hour_sub[0]
      else:
         vwap_day = sum(average_hour_sub)/sum(volume_hour_sub)
      
      vwap.append(vwap_day)

   return np.array(vwap)


class MyDset(Dataset):

    def __init__(self, data, target, window, future=0):
        self.cnter = 0

        x_vals = []
        y_vals = []
        start = 0 + window 
        end = len(data) - future

        for i in range(start, end):
            x = data[(i-window):i]
            y = target[i:(i+future)]

            x_vals.append(x)
            y_vals.append(y)
        
        tensor_x = torch.Tensor(np.array(x_vals))
        tensor_y = torch.Tensor(np.array(y_vals))
        
        self.data = tensor_x
        self.target = tensor_y
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y
    
    def __len__(self):
        return len(self.data)


def trainRNN(model, loader_train, \
      learning_rate=1e-2, device='cpu',\
         criterion=nn.MSELoss(), print_every = 100, epochs=20):

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    print_every = 100
    for e in range(epochs):

        for t, (x,y) in enumerate(loader_train):

            model.train()  # put model to training mode

            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)

            scores = model(x)
        
            loss = criterion(scores, y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if t % print_every == 0:
                print('Epoch %d Iteration %d, loss = %.8f' % (e, t, loss.item()))
                print(t)

    torch.save(model.state_dict(), 'LSTMvwap.ckpt')