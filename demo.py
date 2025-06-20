# -*- coding: utf-8 -*-
from configuration import args
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from data_loader import TrafficFlowDataset
from model.lstm import MyLSTM
from model.gru import MyGRU
from model.cnn_lstm import CNNLSTM
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from sklearn.utils import shuffle
from configuration import args
from func import nextBatch,mape,drawPlot
import pandas as pd
import warnings
from train import train,test,predict

warnings.filterwarnings('ignore')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__== '__main__':
    train_path = "train_data_sort.csv"
    test_path = "test_set.csv"
    train_dataset = TrafficFlowDataset(train_path, train=True)
    test_dataset = TrafficFlowDataset(test_path, train=False)

    if args.model == "gru":
        model = MyGRU(input_size=6,
                      hidden_size=args.hidden_size,
                      output_size=6,
                      drop_prob=args.drop_prob)
    elif args.model == "lstm":
        model = MyLSTM(input_size=6,
                       hidden_size=args.hidden_size,
                       output_size=6,
                       drop_prob=args.drop_prob)
    elif args.model == "cnnlstm":
        model = CNNLSTM(
                       input_size=6,
                       hidden_size=128,
                       output_size=6,
                       drop_prob=0.5)
    print("loading model {} done".format(args.model))

    optimizer =  optim.Adam(params=model.parameters(),lr=args.lr,weight_decay=1e-5)
    #loss_fn = nn.L1Loss()
    loss_fn = nn.MSELoss()
    if torch.cuda.is_available():
        args.iscuda = True
        model = model.to(device)

    train_loss_list,test_loss_list = [],[]
    train_rmse_list,train_mae_list,train_mape_list = [],[],[]
    test_rmse_list,test_mae_list,test_mape_list = [],[],[]

    train_loss_list,test_loss_list = [],[]
    train_rmse_list,train_mae_list,train_mape_list = [],[],[]
    test_rmse_list,test_mae_list,test_mape_list = [],[],[]
    train_times = 0.0

    #梯度截断
    best_loss = float('inf')  # 初始最优 loss 为正无穷
    patience = 50             # 最多容忍多少个 epoch 没有提升
    delta = 1e-4              # 认为有提升的最小 loss 降低幅度
    wait = 0                  # 当前等待次数

    #训练
    for i in range(args.epochs):
        print("=========epoch {}=========".format(i + 1))
        train_rmse,train_mae,train_mape,train_loss = train(model,train_dataset,loss_fn,optimizer)
        train_loss_list.append(train_loss)
        train_rmse_list.append(train_rmse)
        train_mae_list.append(train_mae)
        train_mape_list.append(train_mape)
        print('Epoch: {}, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f},Train Loss: {:.8f}'.format(
            i + 1,train_rmse,train_mae,train_mape,train_loss))
        # eval
        test_rmse,test_mae,test_mape,test_loss = test(model,test_dataset,loss_fn)
        test_loss_list.append(test_loss)
        test_rmse_list.append(test_rmse)
        test_mae_list.append(test_mae)
        test_mape_list.append(test_mape)
        print('Epoch: {}, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f},Test Loss: {:.8f}'.format(
            i + 1,test_rmse,test_mae,test_mape,test_loss,train_loss,test_loss))
        
        # Early Stopping 检查
        # if test_loss + delta < best_loss:
        #     best_loss = test_loss
        #     wait = 0
        #     torch.save({
        #     'model_state_dict': model.state_dict(),
        #     'min_val': train_dataset.min_val,
        #     'max_val': train_dataset.max_val
        # }, 'model.pth')
        #     print(">>> Test loss improved, model saved.")
        # else:
        #     wait += 1
        #     print(">>> No improvement for {} epochs.".format(wait))
        #     if wait >= patience:
        #         print(">>> Early stopping triggered at epoch {}.".format(i + 1))
        #         break
    #保存模型
    #torch.save(model.state_dict(), 'model_state.pth')
    #保存日志
    with open("log.txt","a") as fp:
        fp.write("model:{}, lr: {}, epochs:{}, batch size:{}, hidden_size:{}, train times:{}\n".format(
            args.model,args.lr,args.epochs,args.batch_size,args.hidden_size,train_times))
    #计算均方误差
    metrics = [train_loss_list,test_loss_list,train_rmse_list,test_rmse_list,
        train_mae_list,test_mae_list,train_mape_list,test_mape_list]
    # metrics curve
    fname = "{}_lr{}_b{}_h{}_d{}_metrics.png".format(args.model,args.lr,
        args.batch_size,args.hidden_size,args.drop_prob)
    drawPlot(metrics,fname,["loss","rmse","mae","mape"]) 

    #推理评估模型效果
    predicted_flow = predict(model, test_dataset)

    # 保存预测结果
    df_pred = pd.DataFrame(predicted_flow)
    df_pred.to_csv("predicted_traffic_CNNLSTM.csv", index=False, header=False)
    #df_pred.to_excel("predicted_traffic_excel.xlsx", index=False, header=False, engine="openpyxl")
    print("预测完成，结果已保存到 predicted_traffic_flow.csv")


    # 保存训练/测试 loss 结果
    loss_df = pd.DataFrame({
        "train_loss": train_loss_list,
        "test_loss": test_loss_list,
        "train_rmse": train_rmse_list,
        "test_rmse": test_rmse_list,
        "train_mae": train_mae_list,
        "test_mae": test_mae_list,
        "train_mape": train_mape_list,
        "test_mape": test_mape_list
    })
    loss_df.to_csv(f"{args.model}_loss_metrics.csv", index=False)
    print(f">>> 模型 {args.model} 的训练损失记录已保存到 {args.model}_loss_metrics.csv")

