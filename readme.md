'''
##项目说明
#本项目由github上面的一个项目改编而成，后面是下载地址，之后我可能还会将此项目发在CSDN上面的blog和GitHub上，这是我的blog链接

[纽约项目的下载地址](https://github.com/Yankun168/NYCtrafficFlowPrediction "最新版本下载")


[我的博客地址](https://blog.csdn.net/m0_75237457?spm=1000.2115.3001.5343 "最新版本下载")
## 代码环境要求

- **Python**: 3.12
- **PyTorch**: >=2.2.1
- 其他依赖包详见 `deep-storm3d.yaml` 文件

### 环境安装指南

#### 懒人安装法（一键安装所有依赖）
```bash
#用下面给的这个命令安装
#conda env create -f environment.yaml



此代码是根据GitHub上纽约车流量上的代码改编而成的，主要用于绘制预测结果的对比图。，对于不同的模型，预测结果的对比图可以帮助我们直观地看到各个模型的预测效果。
但是如果仅仅只是绘制对比图，可能会导致图表过于拥挤，难以区分各个模型的预测结果。因此，在绘制对比图时，我们需要注意以下几点：
1. 选择合适的颜色和线型，以便区分不同模型的预测结果。
2. 在图表中添加图例，以便读者能够清楚地知道每条曲线代表哪个模型的预测结果。
3. 调整图表的布局和大小，以便更好地展示数据。

# 代码中使用了pandas和matplotlib库来处理数据和绘制图表。
# pandas用于读取和处理CSV文件中的数据，matplotlib用于绘制图表。
# 代码中还使用了pylab库来设置中文显示和负号显示，以便更好地展示图表。

#请读者慎用我的代码，并根据自己的需求进行修改。
#谨记：
# 1. 代码仅供参考，切忌不可直接复制粘贴使用。
'''