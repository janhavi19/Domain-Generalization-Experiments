from model.multitaskmodel import MultitaskBayesianNN
from dataloaderr.dataloading import MultitaskDataset
import torch.optim as optim
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pickle
import zarr

def replace_y(x):
    mapping = {1: 0, 2: 1, 3: 2, 4: 3}
    return mapping.get(x, x)

def replace_d(x):
    mapping = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9, 13: 10, 14: 11, 15: 12, 16: 13, 17: 14}
    return mapping.get(x, x)

def loso(X,y,d):
    left_out_subject = 14
    idx = (d != left_out_subject)
    X_train = X[idx]
    y_train = y[idx]
    d_train = d[idx]

    # test data selecting just 14 subject
    idxt = (d == left_out_subject)
    X_test = X[idxt]
    y_test = y[idxt]
    d_test = d[idxt]


    return X_train, y_train, d_train, X_test, y_test, d_test

def data_loader():
    zarr_array = zarr.open("./dataset/chest_ECG_w60_mw60_ts70_cl10_cs1_fp[1.0].zarr/", mode="r")
    signal = zarr_array['raw_signal'][:]
    target_all = zarr_array['target'][:]
    subjects_all = zarr_array['subject'][:]

    SUBJECTS_IDS = list(range(2, 18))
    subjects = SUBJECTS_IDS[:]
    classes = [1, 2, 3, 4]

    subset_map = [
        idx
        for idx, i in enumerate(target_all)
        if i in classes and subjects_all[idx] in subjects
    ]

    idx = subset_map
    X = signal[idx]
    y = target_all[idx]
    d = subjects_all[idx]

    y_updated = np.vectorize(replace_y)(y)
    d_updated = np.vectorize(replace_d)(d)st
    #train-test splitting by leaving one subject out for test
    X_train, y_train, d_train, X_test, y_test, d_test = loso(X,y_updated,d_updated)
    

    multitask_data = MultitaskDataset(X_train, y_train, d_train, X_test, y_test, d_test)
    trainloader = multitask_data.train_loader(batch_size=32, shuffle=False)
    testloader = multitask_data.test_loader(batch_size=32, shuffle=False)

    return  trainloader,testloader


if __name__ == '__main__':

    trainloader, testloader = data_loader()
    # define your model, input/output/domain dimensions, and number of tasks
    model = MultitaskBayesianNN(700, 7, 14)

    # defining loss function (e.g. cross entropy) and optimizer (e.g. Adam)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    print_every = 10
    n_batches = len(trainloader)
    start_time = time.time()

    # training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, batch in enumerate(trainloader):
            inputs, labels, domains = batch
            optimizer.zero_grad()
            inputs = inputs.unsqueeze(1)
            inputs = inputs.transpose(1, 2)  # Transpose the input so that sequence length is the last dimension
            inputs = inputs.view(inputs.shape[0], 1, inputs.shape[1], inputs.shape[2])

            outputs = model(inputs, labels, domains)

            labels = labels.squeeze()

            print(labels.min(), labels.max())
            loss = criterion(outputs[:, :3], labels)+criterion(outputs[:, 14:], domains)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / n_batches
        print("Epoch {}, train_loss: {:.5f} took: {:.2f}s".format(epoch + 1, epoch_loss, time.time() - start_time))
        running_loss = 0.0
        start_time = time.time()
        # Save the model state dict
        torch.save(model.state_dict(), './results/model_weights.pth')

        # Save the entire model (including architecture and state dict)
        torch.save(model, './results/model.pth')

