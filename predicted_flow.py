import torch
import pandas as pd
from model.lstm import MyLSTM
from data_loader import TrafficFlowDataset
from train import predict

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载数据
test_path = "test_set.csv"
test_dataset = TrafficFlowDataset(test_path, train=False)

# 初始化模型（确保结构和训练时一致）
model = MyLSTM(input_size=6, hidden_size=64, output_size=6, drop_prob=0.5)

# 加载模型参数
model.load_state_dict(torch.load("model_state.pth", map_location=device))

# 设置为评估模式并移动到相应设备
model.eval()
model.to(device)

# 推理
predicted_flow = predict(model, test_dataset)

# 保存结果
df_pred = pd.DataFrame(predicted_flow)
df_pred.to_csv("predicted_traffic_pre.csv", index=False, header=False)

print("预测完成，结果已保存到 predicted_traffic_pre.csv")
