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


class RNN(nn.Module):
    '''Implement the LSTM model for forecasting.'''
    def __init__(self, input_size, hidden_size, \
        num_layer, num_class = 1):

        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.num_class = num_class

        self.lstm = nn.LSTM(input_size, hidden_size, num_layer, \
            batch_first = True) # initial hidden and cell state default 0

        self.fc1 = nn.Linear(hidden_size, num_class*2)
        self.fc2 = nn.Linear(num_class*2, num_class)

    def forward(self, input):
        
        _out , (h_out, _) = self.lstm(input) # output of LSTM is out,(hidden,cell)

        r = h_out.view(-1, self.hidden_size) # reshape to use linear for prediction

        output = self.fc1(r)
        output = F.tanh(output)

        output = self.fc2(output)
        output = F.tanh(output)

        return output