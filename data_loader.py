import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq 


def sliding_window(seqs, size):
    """
    Apply sliding window to 3D tensor (time, feature) -> (samples, window, feature)
    """
    result = []
    for i in range(seqs.shape[0] - size + 1):
        result.append(seqs[i:i + size])
    return np.array(result)


class TrafficFlowDataset:
    def __init__(self, csv_path, window_size=7, train=True, split_ratio=0.8):
        self.window_size = window_size
        self.train = train
        self.split_ratio = split_ratio

        # 预处理
        self.data = self._load_and_preprocess(csv_path)
        self.max_val = self.data.max(axis=0)  # shape: (6,)
        self.min_val = self.data.min(axis=0)
        self.data = (self.data - self.min_val) / (self.max_val - self.min_val)

        # Apply sliding window
        sequences = sliding_window(self.data, window_size)

        # 分离训练集合和测试集
        split_index = int(len(sequences) * self.split_ratio)
        if self.train:
            self.data = torch.from_numpy(sequences[:split_index]).float()
        else:
            self.data = torch.from_numpy(sequences[split_index:]).float()

    def _load_and_preprocess(self, csv_path):
        df = pd.read_csv(csv_path)

        # 选取6个车流量相关字段
        features = ['微型车流量', '中型车流量', '大车流量', '长车流量', '轻型车流量', '车流量']
        data = df[features].values  # shape: (时间, 6)
        return data
    #反归一化
    def denormalize(self, x):
        return x * (self.max_val - self.min_val) + self.min_val


if __name__ == "__main__":
    csv_path = "data/train_data.csv"
    train_dataset = TrafficFlowDataset(csv_path, train=True)
    test_dataset = TrafficFlowDataset(csv_path, train=False)

    print("Train shape:", train_dataset.data.shape)  # (样本数, 窗口长度, 特征数)
    print("Test shape:", test_dataset.data.shape)



