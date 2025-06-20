import pandas as pd
from sklearn.model_selection import train_test_split

# 提取的字段列表
selected_columns = ['微型车流量', '中型车流量', '大车流量', '长车流量', '轻型车流量', '车流量']

# 读取CSV数据
data = pd.read_csv('train_data_sort.csv', usecols=selected_columns,encoding='utf-8')

# 划分训练集和测试集
train_data, test_data = train_test_split(data, 
                                       test_size=0.2,
                                       random_state=42,
                                       shuffle=False)

# 模型大小
train_data.to_csv('train_set.csv', index=False)
test_data.to_csv('test_set.csv', index=False)
