import pandas as pd
import numpy as np
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 设置字体
plt.rcParams["font.sans-serif"] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据并划分训练集/测试集
df = pd.read_excel('data_selected_features.xlsx', sheet_name='Sheet1').dropna()
df1 = pd.read_excel('data_without_some_cols.xlsx', sheet_name='Sheet1').dropna()
X = df
y = df1["CO2 adsorption uptake (mol/g)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 合并训练集特征和目标变量
train_data = pd.concat([X_train, y_train], axis=1)

# 自动推断数据元数据
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_data)


# 计算标准差和方差的差异
def calculate_diff(original_data, synthetic_data):
    std_diff = np.abs(original_data.std() - synthetic_data.std())
    var_diff = np.abs(original_data.var() - synthetic_data.var())
    return std_diff, var_diff


# 定义评估函数（用于标准差和方差对比）
def evaluate_with_stats(seed):
    # 初始化TVAE模型
    tvae_model = TVAESynthesizer(
        embedding_dim=128,
        compress_dims=(128, 64),
        decompress_dims=(64, 128),
        epochs=1000,
        verbose=False,
        metadata=metadata,
        # random_state=seed
    )
    # 训练模型
    tvae_model.fit(train_data)  # 使用包含目标变量的完整数据
    # 生成合成数据
    synthetic_data = tvae_model.sample(len(X_train))

    # 计算标准差和方差差异
    std_diff_all = []
    var_diff_all = []

    for col in X_train.columns:
        std_diff, var_diff = calculate_diff(X_train[col], synthetic_data[col])
        std_diff_all.append(std_diff)
        var_diff_all.append(var_diff)

    return np.sum(std_diff_all), np.sum(var_diff_all)


# 设置种子列表
seeds = [42, 123, 10, 99, 200]

# 存储每个种子的标准差和方差差异
results = {}

for seed in seeds:
    std_diff, var_diff = evaluate_with_stats(seed)
    results[seed] = {'Std_Diff': std_diff, 'Var_Diff': var_diff}

# 打印每个种子的评估结果
for seed, metrics in results.items():
    print(f"种子 {seed}: 标准差差异 = {metrics['Std_Diff']:.4f}, 方差差异 = {metrics['Var_Diff']:.4f}")

# 选择最好的种子（标准差和方差差异之和最小）
best_seed = min(results, key=lambda x: results[x]['Std_Diff'] + results[x]['Var_Diff'])
print(
    f"\n最好的种子是: {best_seed}，对应的标准差差异 = {results[best_seed]['Std_Diff']:.4f}, 方差差异 = {results[best_seed]['Var_Diff']:.4f}")
