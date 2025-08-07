import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei'] # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 数据准备
metrics = ['MSE', 'MAE', 'R2']
models = ['原始数据', 'TVAE-RF', 'CTGAN-RF']
scores = np.array([
    [0.4345, 0.4504, 0.6895],  # 原始数据
    [0.3483, 0.4320, 0.7511],  # TVAE-RF
    [0.3247, 0.4069, 0.7680],  # CTGAN-RF
])

# 创建图形
fig, ax = plt.subplots(figsize=(8, 5))

# 每个模型绘制直方图，使用橙色渐变色
width = 0.2  # 每个条形的宽度
x = np.arange(len(metrics))  # X轴的位置

# 使用鲜艳的橙色渐变
gradient_colors = plt.cm.Oranges(np.linspace(0.5, 1, len(models)))  # 使用Oranges颜色映射，调整渐变范围

# 为每个模型绘制直方图并应用不同的渐变色
for i in range(len(models)):
    bars = ax.bar(x + i * width, scores[i, :], width=width, label=models[i], color=gradient_colors[i])

    # 在顶部添加相关数据
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}', ha='center', va='bottom')

# 设置X轴的位置和标签
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)

# 添加标签和标题
ax.set_xlabel('评估指标')
ax.set_ylabel('得分')
ax.set_title('不同机器学习算法的MSE、MAE和R2比较（鲜艳橙色渐变）')

# 去掉网格
ax.grid(False)

# 保留顶部和右边的框线
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

# 添加图例
ax.legend()

# 显示图形
plt.tight_layout()
plt.show()
