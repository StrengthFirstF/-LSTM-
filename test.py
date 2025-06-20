import pandas as pd
import matplotlib.pyplot as plt


########################################绘制模型误差图###################
# 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

predict_length = 270

# 数据准备
df_test = pd.read_csv("test_set.csv").iloc[:predict_length]
df_pred_CNNLSTM = pd.read_csv("预测数据/predicted_traffic_CNNLSTM.csv", header=None).iloc[:predict_length]
df_pred_LSTM = pd.read_csv("预测数据/predicted_traffic_LSTM.csv", header=None).iloc[:predict_length]
df_pred_GRU = pd.read_csv("预测数据/predicted_traffic_GRU.csv", header=None).iloc[:predict_length]

columns = ['微型车流量', '中型车流量', '大车流量', '长车流量', '轻型车流量', '车流量']
df_pred_LSTM.columns = columns
df_pred_CNNLSTM.columns = columns
df_pred_GRU.columns = columns
df_true = df_test[columns]

# 对齐数据
min_len = min(len(df_true), len(df_pred_LSTM), len(df_pred_GRU), len(df_pred_CNNLSTM))
df_true = df_true.iloc[:min_len]
df_pred_LSTM = df_pred_LSTM.iloc[:min_len]
df_pred_GRU = df_pred_GRU.iloc[:min_len]
df_pred_CNNLSTM = df_pred_CNNLSTM.iloc[:min_len]

# 计算误差
error_LSTM = df_pred_LSTM - df_true
error_GRU = df_pred_GRU - df_true
error_CNNLSTM = df_pred_CNNLSTM - df_true

# 创建画布
plt.figure(figsize=(20, 15))

# 绘制六个子图
for idx, col in enumerate(columns, 1):
    plt.subplot(3, 2, idx)  # 3行2列布局
    
    # 绘制误差曲线
    plt.plot(error_LSTM[col].values,
             label="LSTM误差", 
             linestyle='--',
             linewidth=1.8,
             color='red',
             alpha=0.8)

    plt.plot(error_GRU[col].values,
             label="GRU误差", 
             linestyle=':',
             linewidth=1.8,
             color='blue',
             alpha=0.8)

    plt.plot(error_CNNLSTM[col].values,
             label="CNNLSTM误差", 
             linestyle='-.',
             linewidth=1.8,
             color='green',
             alpha=0.8)

    # 基准线
    plt.axhline(0, color='gray', linestyle=':', linewidth=1)

    # 图表装饰
    plt.title(f"{col}预测误差对比", fontsize=12, pad=15)
    plt.xlabel("时间步（单位：天×48）", fontsize=10)
    plt.ylabel("误差（辆/30min）", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper right', fontsize=9)

    # 坐标轴优化
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

# 调整布局并保存
plt.tight_layout(pad=3.0)
plt.savefig('预测误差对比图.png', dpi=300, bbox_inches='tight')
plt.show()
