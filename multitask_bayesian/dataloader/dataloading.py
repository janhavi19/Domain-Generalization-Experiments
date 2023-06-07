import zarr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pickle
import zarr
import torch
from torch.utils.data import TensorDataset, DataLoader

import zarr
import numpy as np


class MultitaskDataset():
    def __init__(self, X_train, y_train, d_train, X_test, y_test, d_test):
        self.X_train = X_train
        self.y_train = y_train
        self.d_train = d_train
        self.X_test = X_test
        self.y_test = y_test
        self.d_test = d_test

    def train_loader(self, batch_size, shuffle=True):
        X = torch.from_numpy(self.X_train).squeeze().float()
        y = torch.from_numpy(self.y_train).squeeze().long()
        d = torch.from_numpy(self.d_train.astype(np.int64)).squeeze().long()

        dataset = TensorDataset(X, y, d)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def test_loader(self, batch_size, shuffle=True):
        Xt = torch.from_numpy(self.X_test).squeeze().float()
        yt = torch.from_numpy(self.y_test).squeeze().long()
        dt = torch.from_numpy(self.d_test.astype(np.int64)).squeeze().long()

        dataset = TensorDataset(Xt, yt, dt)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
