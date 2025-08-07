import pandas as pd
import numpy as np
import torch
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import gaussian_kde
from sdv.evaluation.single_table import evaluate_quality
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import random


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 设置字体
plt.rcParams["font.sans-serif"] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据并划分训练集/测试集
df = pd.read_excel('data_without_some_cols.xlsx', sheet_name='Sheet1').dropna()
high_variability_columns = ['SSABD', 'SSAAD', 'MW', 'PPM', 'LA']
df[high_variability_columns] = df[high_variability_columns].apply(lambda x: np.log(x + 1))
X = df.drop("CO2 uptake", axis=1)
y = df["CO2 uptake"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 合并训练集特征和目标变量
train_data = pd.concat([X_train, y_train], axis=1)

# 自动推断数据元数据
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_data)

# 初始化TVAE模型
tvae_model = TVAESynthesizer(
    embedding_dim=128,  # 隐变量维度
    compress_dims=(128, 64),  # 编码器压缩层
    decompress_dims=(64, 128),  # 解码器解压缩层
    epochs=1000,  # 增加训练轮次
    verbose=True,
    metadata=metadata,
    # random_state=42
)

# 训练模型
tvae_model.fit(train_data)

# 生成合成数据
synthetic_data = tvae_model.sample(len(train_data))

features_col = df.columns.tolist()


def plot_kde(data, color, label):
    kde = gaussian_kde(data)
    x = np.linspace(min(data), max(data), 1000)
    plt.plot(x, kde(x), color=color, linewidth=2, label=label)

# 对比目标变量分布
for col in features_col:
    plt.figure(figsize=(10, 4))
    plt.hist(train_data[col], bins=20, alpha=0.5, label='原始数据')
    plt.hist(synthetic_data[col], bins=20, alpha=0.5, label='TVAE生成数据')
    plt.xlabel(f"{col}")
    plt.ylabel("频率")
    plt.legend()
    plt.title(f"原始数据与生成数据{col}分布对比")
    plt.show()

for col in features_col:
    plt.figure(figsize=(10, 4))
    plot_kde(train_data[col], color='blue', label='原始数据分布')
    plot_kde(synthetic_data[col], color='orange', label='生成数据分布')
    plt.title(f"{col}的原始数据与生成数据分布对比")
    plt.legend()
    plt.show()

print("原始数据每列的均值与标准差：\n")
for col in train_data.columns:
    mean_val = train_data[col].mean()
    std_val = train_data[col].std()
    print(f"{col}：均值 = {mean_val:.4f}, 标准差 = {std_val:.4f}")

print("\n合成数据每列的均值与标准差：\n")
for col in synthetic_data.columns:
    mean_val = synthetic_data[col].mean()
    std_val = synthetic_data[col].std()
    print(f"{col}：均值 = {mean_val:.4f}, 标准差 = {std_val:.4f}")

# 计算原始数据与合成数据的Spearman相关系数差异
original_corr = train_data.corr(method='spearman')
plt.figure(figsize=(8, 6))
sns.heatmap(original_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.xticks(rotation=45)  # 旋转x轴标签
plt.yticks(rotation=0)
plt.title("原始数据相关性矩阵")
plt.show()

synthetic_corr = synthetic_data.corr(method='spearman')
plt.figure(figsize=(8, 6))
sns.heatmap(synthetic_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.xticks(rotation=45)  # 旋转x轴标签
plt.yticks(rotation=0)
plt.title("生成数据相关性矩阵")
plt.show()

# 绘制相关系数差异图
plt.figure(figsize=(8, 6))
plt.scatter(original_corr, synthetic_corr, alpha=0.5)
plt.plot([-1, 1], [-1, 1], 'k--')  # 对角线
plt.xlabel("原始数据相关性")
plt.ylabel("合成数据相关性")
plt.title("特征相关性一致性验证")
plt.show()


# 分离合成数据的特征和目标
X_syn = synthetic_data.drop("CO2 uptake", axis=1)
y_syn = synthetic_data["CO2 uptake"]

# 合并原始和合成数据
X_train_augmented = pd.concat([X_train, X_syn], axis=0)
y_train_augmented = pd.concat([y_train, y_syn], axis=0)

# 定义评估函数
def evaluate_performance(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=200,
                                  random_state=42,
                                  min_samples_split=2,
                                  # max_depth=10
                                  )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "MSE": mean_squared_error(y_test, y_pred),
        "R²": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred)
    }

# 评估原始数据
original_metrics = evaluate_performance(X_train, y_train, X_test, y_test)

# 评估TVAE增强数据
tvae_metrics = evaluate_performance(X_train_augmented, y_train_augmented, X_test, y_test)

print("生成数据与真实数据的相似得分")
print(f'生成数据与真实数据的相似得分：{evaluate_quality(synthetic_data, train_data, metadata=metadata)}')

# 输出结果
print("原始数据性能:")
print(f"MSE = {original_metrics['MSE']:.4f}, R² = {original_metrics['R²']:.4f}, MAE = {original_metrics['MAE']:.4f}")

print("\nTVAE增强数据性能:")
print(f"MSE = {tvae_metrics['MSE']:.4f}, R² = {tvae_metrics['R²']:.4f}, MAE = {tvae_metrics['MAE']:.4f}")

def plot_pca(X_original, X_synthetic, title):
    # 标准化数据
    scaler = StandardScaler()
    X_original_scaled = scaler.fit_transform(X_original)
    X_synthetic_scaled = scaler.transform(X_synthetic)

    # 使用PCA将数据降至二维
    pca = PCA(n_components=2)
    X_original_pca = pca.fit_transform(X_original_scaled)
    X_synthetic_pca = pca.transform(X_synthetic_scaled)

    # 绘制PCA图
    plt.figure(figsize=(10, 6))
    plt.scatter(X_original_pca[:, 0], X_original_pca[:, 1], alpha=0.5, label="原始数据", color='blue')
    plt.scatter(X_synthetic_pca[:, 0], X_synthetic_pca[:, 1], alpha=0.5, label="生成数据", color='orange')
    plt.title(f"PCA：{title}")
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.legend()
    plt.show()

# 比较原始数据和生成数据的PCA
plot_pca(X, X_syn, "原始数据与生成数据PCA对比")