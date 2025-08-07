import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import random

# 设置字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei'] # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

"""读取数据，删除相关的行"""
data = pd.read_excel('data.xlsx')
# data = data.drop(columns="Type of amine")

# 进行数值列的正态性检验和相关性分析
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

# 对每一个数据进行shapiro-wilk正态性检验
normality_results = {col: stats.shapiro(data[col].dropna()) for col in numeric_columns}

# 计算相关性矩阵
corr_matrix = data[numeric_columns].corr(method='spearman')
corr_matrix2 = data[numeric_columns].corr(method='pearson')

# 设置相关性阈值
correlation_threshold = 0.7

# 查找高度相关的变量并删除
high_correlation_pairs = []
for col1 in corr_matrix.columns:
    for col2 in corr_matrix.columns:
        if col1 != col2 and abs(corr_matrix[col1][col2]) >= correlation_threshold:
            high_correlation_pairs.append((col1, col2))

# 找到并删除一个高相关性的变量
cols_to_drop = set()
for col1, col2 in high_correlation_pairs:
    cols_to_drop.add(col1) # 删除第二个变量，可以根据需求选择删除任意一个

# 删除高相关性变量
# list_col_drop = list(cols_to_drop)
data_cleaned = data.drop(columns=cols_to_drop)

# 重新计算VIF
X = data_cleaned[numeric_columns.difference(cols_to_drop)]
X_const = add_constant(X)

# 计算VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

# 输出删除后的数据和VIF值
print("删除的高相关性变量:")
print(cols_to_drop)
print("\n新的VIF值:")
print(vif_data.sort_values(by="VIF", ascending=False))

# 打印正态性检验结果和相关性矩阵
print("正态性检验结果:")
for col, result in normality_results.items():
    print(f"{col}: Statistic = {result.statistic}, p-value = {result.pvalue}")

# 输出VIF结果
print("\n方差膨胀因子（VIF）结果:")
print(vif_data.sort_values(by="VIF", ascending=False))

plt.figure(figsize=(8, 6))
# 绘制斯皮尔曼相关性矩阵热图
sns.heatmap(corr_matrix2, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.xticks(rotation=45)  # 旋转x轴标签
plt.yticks(rotation=0)
# 设置标题和显示
plt.title('皮尔逊相关矩阵')
plt.show()

plt.figure(figsize=(8, 6))
# 绘制斯皮尔曼相关性矩阵热图
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.xticks(rotation=45)  # 旋转x轴标签
plt.yticks(rotation=0)
# 设置标题和显示
plt.title('斯皮尔曼相关矩阵')
plt.show()

# CO2吸附量的分布与箱线图
plt.figure(figsize=(10, 4))
sns.histplot(data["CO2 uptake"], kde=True)
plt.title("CO2吸附量分布")
plt.show()

sns.boxplot(x=data["CO2 uptake"])
plt.title("CO2吸附量箱线图")
plt.show()

# 加载量（LA）与CO2吸附量的散点图
sns.scatterplot(data=data, x="LA", y="CO2 uptake")
plt.title("加载量 vs CO2吸附量")
plt.show()

# 温度与CO2吸附量的箱线图
sns.boxplot(data=data, x="Temp", y="CO2 uptake")
plt.title("温度对CO2吸附量的影响")
plt.xticks(rotation=45)
plt.show()

data_cleaned.to_excel("data_without_some_cols.xlsx", index=False)