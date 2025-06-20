import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl

# 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


predict_length = 270
# 数据准备
df_test = pd.read_csv("test_set.csv")
df_test = df_test.iloc[:predict_length]  
df_pred_CNNLSTM = pd.read_csv("预测数据/predicted_traffic_CNNLSTM.csv", header=None)
df_pred_LSTM = pd.read_csv("预测数据/predicted_traffic_LSTM.csv", header=None)
df_pred_GRU = pd.read_csv("预测数据/predicted_traffic_GRU.csv", header=None)

#截取数据
df_pred_LSTM  = df_pred_LSTM .iloc[:predict_length]
df_pred_CNNLSTM = df_pred_CNNLSTM.iloc[:predict_length]
df_pred_GRU= df_pred_GRU.iloc[:predict_length]

columns = ['微型车流量', '中型车流量', '大车流量', '长车流量', '轻型车流量', '车流量']
df_pred_LSTM.columns = columns
df_pred_CNNLSTM.columns = columns
df_pred_GRU.columns = columns
df_true = df_test[columns]

# 数据对齐
#LSTM
min_len_LSTM = min(len(df_true), len(df_pred_LSTM))
df_true_LSTM = df_true.iloc[:min_len_LSTM]
df_pred_LSTM = df_pred_LSTM.iloc[:min_len_LSTM]
#CNNLSTM
min_len_CNNLSTM = min(len(df_true), len(df_pred_CNNLSTM))
df_true_CNNLSTM = df_true.iloc[:min_len_CNNLSTM]
df_pred_CNNLSTM = df_pred_CNNLSTM.iloc[:min_len_CNNLSTM]

min_len_GRU = min(len(df_true), len(df_pred_GRU))
df_true_GRU = df_true.iloc[:min_len_GRU]
df_pred_GRU = df_pred_GRU.iloc[:min_len_GRU]

# 创建画布
plt.figure(figsize=(20, 15))

# 绘制六个子图
for idx, col in enumerate(columns, 1):
    plt.subplot(3, 2, idx)  # 3行2列布局
    
    # 绘制曲线
    plt.plot(df_true[col].values, 
            label="真实值", 
            linewidth=1.5,
            color='#1f77b4',
            marker='o',
            markersize=3,
            markevery=20)
    
    plt.plot(df_pred_LSTM[col].values, 
            label="LSTM预测值", 

            linestyle='--',
            linewidth=1.8,
            color='#ff7f0e',
            alpha=0.8)
    plt.plot(df_pred_GRU[col].values, 
            label="GRU预测值", 

            linestyle=":",
            linewidth=1.8,
            color='#2ca02c',
            alpha=0.8)
    plt.plot(df_pred_CNNLSTM[col].values, 
            label="CNNLSTM预测值", 

            linestyle='-.',
            linewidth=1.8,
            color='#d62728',
            alpha=0.8)
    
    # 图表装饰
    plt.title(f"{col}对比", fontsize=12, pad=15)
    plt.xlabel("单位/天", fontsize=10)
    plt.ylabel("流量值", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper right', fontsize=9)
    
    # 优化坐标轴
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

# 调整布局并保存
plt.tight_layout(pad=3.0)
plt.savefig('全字段对比图.png', dpi=300, bbox_inches='tight')
plt.legend()
plt.tight_layout(pad=5.0)
plt.show()


