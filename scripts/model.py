#!/usr/bin/env python
from __future__ import print_function

##### add python path #####
import sys
import os

from collections import deque
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from sklearn.utils import shuffle

ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)
EPS = 1e-6


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, learning_rate, device, option):
        super(MLP, self).__init__()
        self.device = device
        self.option = option
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.H = len(self.hidden_layers)
        self.fc = nn.ModuleList([])
        self.fc.append(nn.Linear(self.input_dim, self.hidden_layers[0]))
        self.lr = learning_rate
        for i in range(1, self.H):
            if self.option[0] == 'batch-norm':
                self.fc.append(nn.BatchNorm1d(self.hidden_layers[i-1]))
            self.fc.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
        if self.option[0] == 'batch-norm':
                self.fc.append(nn.BatchNorm1d(self.hidden_layers[self.H - 1]))
        self.fc.append(nn.Linear(self.hidden_layers[self.H - 1], self.output_dim))
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        self.loss = nn.MSELoss()

    def forward(self, x):
        # forward network and return
        for i in range(0,self.H):
            if self.option[1] == 'relu':
                x = F.relu(self.fc[i](x))
            if self.option[1] == 'leaky-relu':
                x = F.leaky_relu(self.fc[i](x))
            if self.option[1] == 'sigmoid':
                x = F.sigmoid(self.fc[i](x))
            if self.option[1] == 'tanh':
                x = F.tanh(self.fc[i](x))
            
        x = self.fc[self.H](x)
        if self.option[2] == 'sigmoid':
            x = F.sigmoid(x)
        if self.option[2] == 'tanh':
            x = F.tanh(x)
        return x


    def mse_loss(self, z, action):
        return self.loss(z, action)
