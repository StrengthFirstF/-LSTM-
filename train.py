# -*- coding: utf-8 -*-
from configuration import args
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from sklearn.utils import shuffle
from configuration import args
from func import nextBatch,mape,drawPlot
import pandas as pd
import math


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model,train_dataset,loss_fn,optimizer):
    model.train()
    iteration = 0
    for batch in nextBatch(shuffle(train_dataset.data),args.batch_size):
        x,y = batch[:,:-1,:], batch[:,-1,:]
        if args.iscuda:
            x,y = x.to(device),y.to(device)
        y_hat = model(x)
        l = loss_fn(y_hat, y)
        optimizer.zero_grad(set_to_none=True)
        l.backward()
        # # 梯度裁剪
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
        optimizer.step()
        iteration += 1
        if iteration % 10 == 0:
            print("Iteraion {}, Train Loss: {:.8f}".format(iteration, l.item()))
    
    train_rmse,train_mae,train_mape,train_loss = test(model,train_dataset,loss_fn)
    
    return (train_rmse,train_mae,train_mape,train_loss)

def test(model,test_dataset,loss_fn):
    model.eval()
    y_hats = []
    test_l_sum,c = 0,0
    with torch.no_grad():
        for batch in nextBatch(test_dataset.data, batch_size=args.batch_size):
            x,y = batch[:,:-1,:], batch[:,-1,:]
            if args.iscuda:
                x,y = x.to(device),y.to(device)
            y_hat = model(x)
            test_l_sum += loss_fn(y_hat,y).item()
            c += 1
            y_hats.append(y_hat.detach().cpu().numpy())
        y_hats = np.concatenate(y_hats)
    y_true = test_dataset.data[:,-1,:]
    y_hats = test_dataset.denormalize(y_hats)
    y_true = test_dataset.denormalize(y_true)
    y_true = y_true.reshape(y_true.size(0),-1)
    rmse_score,mae_score,mape_score = math.sqrt(mse(y_true, y_hats)), \
                                mae(y_true, y_hats), \
                                mape(y_true, y_hats)    
    return (rmse_score,mae_score,mape_score,test_l_sum / c)


def predict(model, dataset):
    """使用训练好的模型对测试数据进行预测"""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in nextBatch(dataset.data, batch_size=args.batch_size):
            x = batch[:, :-1, :]
            if args.iscuda:
                x = x.to(device)
            y_hat = model(x)
            predictions.append(y_hat.detach().cpu().numpy())

    predictions = np.concatenate(predictions)
    predictions = dataset.denormalize(predictions)  # 反归一化
    return predictions