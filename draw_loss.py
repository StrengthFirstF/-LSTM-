import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
metrics = ['test_loss', 'test_rmse', 'test_mae', 'test_mape']
titles = ['Loss', 'RMSE', 'MAE', 'MAPE']

models = ['lstm', 'gru', 'cnnlstm']
colors = ['#1f77b4', '#2ca02c', '#d62728']
linestyles = ['-', '--', '-.']

for i, (ax, metric, title) in enumerate(zip(axs.ravel(), metrics, titles)):
    for model, color, ls in zip(models, colors, linestyles):
        df = pd.read_csv(f"{model}_loss_metrics.csv")
        ax.plot(df[metric], label=f"{model.upper()}", color=color, linestyle=ls)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.grid(True, linestyle=':', alpha=0.6)
    #if i == 0:
    ax.legend()

plt.tight_layout()
plt.savefig("模型性能多指标对比图.png", dpi=300)
plt.show()

