import pandas as pd
import matplotlib.pyplot as plt

#设置中文显示
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def draw_population(csv_path):


    # 读取数据
    df = pd.read_csv(csv_path)
    df['数据时间'] = pd.to_datetime(df['数据时间'].str.strip())  # 去除空白并转为时间格式
    df = df.sort_values('数据时间')

    # 提取六个车流量字段
    flow_columns = ['微型车流量', '中型车流量', '大车流量', '长车流量', '轻型车流量', '车流量']

    # 绘制车流量随时间变化的曲线
    plt.figure(figsize=(16, 8))
    offset = 5000  # 每条线之间的间隔
    for idx, col in enumerate(flow_columns):
        plt.plot(df['数据时间'], df[col] + idx * offset, label=col)


    plt.title("6类车流量时间变化曲线", fontsize=16)
    plt.xlabel("时间")
    plt.ylabel("车流量")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

if __name__ == '__main__':
    csv_path = "train_data_sort.csv"
    draw_population(csv_path)