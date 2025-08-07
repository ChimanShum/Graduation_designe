import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 数据
metrics = ['MSE', 'MAE', 'R2']
models = ['线性回归', '决策树', '随机森林', '梯度提升树', '支持向量机']
scores = np.array([
    [0.6108, 0.5514, 0.4967],  # 线性回归
    [0.7176, 0.5574, 0.4086],  # 决策树
    [0.4345, 0.4504, 0.6895],  # 随机森林
    [0.7360, 0.5485, 0.3935],  # 梯度提升树
    [1.1075, 0.6538, 0.0873],  # 支持向量机
])

# 创建图形
fig, ax = plt.subplots(figsize=(8, 5))

# 为每个模型绘制直方图，使用渐变色
width = 0.15  # 每个条形的宽度
x = np.arange(len(metrics))  # X轴的位置

# 使用渐变颜色
for i in range(len(models)):
    # 创建一个渐变色，从浅到深
    gradient_colors = plt.cm.viridis(np.linspace(0, 1, len(models)))  # 使用viridis颜色映射
    ax.bar(x + i * width, scores[i, :], width=width, label=models[i], color=gradient_colors[i])

# 设置X轴的位置和标签
ax.set_xticks(x + width * 2)
ax.set_xticklabels(metrics)

# 添加标签和标题
ax.set_xlabel('评估指标')
ax.set_ylabel('得分')
ax.set_title('不同机器学习算法的MSE、MAE和R2比较')

# 添加图例
ax.legend()

# 显示图形
plt.tight_layout()
plt.show()
