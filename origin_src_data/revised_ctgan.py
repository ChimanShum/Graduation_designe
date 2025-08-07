import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sdv.single_table import CTGANSynthesizer  # 使用CTGAN生成表格数据
from sdv.metadata import SingleTableMetadata
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import gaussian_kde
import random
from sklearn.model_selection import cross_val_score

# 设置随机种子，确保结果可复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 设置字体
plt.rcParams["font.sans-serif"] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取原始数据
df = pd.read_excel('data_without_some_cols.xlsx', sheet_name='Sheet1').dropna()

# 划分原始数据为训练集和测试集（保留测试集不参与增强）
X = df.drop("CO2 adsorption uptake (mol/g)", axis=1)
y = df["CO2 adsorption uptake (mol/g)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放（标准化）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 合并训练集的特征和目标变量
train_data = pd.concat([pd.DataFrame(X_train_scaled, columns=X.columns), y_train], axis=1)

# 自动推断数据元数据
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_data)

# 保存元数据（可选但推荐）
# metadata.save_to_json('metadata.json')

# 初始化CTGAN模型（增加训练周期）
model = CTGANSynthesizer(epochs=5000,  # 增加训练周期
                         verbose=True,
                         metadata=metadata,
                         batch_size=32,  # 设置适当的batch_size
                         generator_lr=0.001,  # 设置生成器学习率
                         discriminator_lr=0.001)  # 设置判别器学习率
model.fit(train_data)

# 生成与原始训练集相同数量的合成数据
synthetic_data = model.sample(len(train_data))

# 分离合成数据的特征和目标变量
X_syn = synthetic_data.drop("CO2 adsorption uptake (mol/g)", axis=1)
y_syn = synthetic_data["CO2 adsorption uptake (mol/g)"]

# 合并原始训练集和合成数据
X_train_augmented = pd.concat([pd.DataFrame(X_train_scaled, columns=X.columns), X_syn], axis=0)
y_train_augmented = pd.concat([y_train, y_syn], axis=0)

# 特征列
features_col = train_data.columns.tolist()

# KDE绘图函数
def plot_kde(data, color, label):
    kde = gaussian_kde(data)
    x = np.linspace(min(data), max(data), 1000)
    plt.plot(x, kde(x), color=color, label=label, linewidth=2)

# 绘制每个特征的分布对比
for col in features_col:
    plt.figure(figsize=(10, 4))
    plt.hist(train_data[col], bins=20, alpha=0.5, label='原始数据')
    plt.hist(synthetic_data[col], bins=20, alpha=0.5, label='合成数据')
    plt.xlabel(f"{col}")
    plt.ylabel("频率")
    plt.legend()
    plt.title(f"{col}的原始数据与生成数据分布对比")
    plt.show()

for col in features_col:
    plot_kde(train_data[col], color='blue', label='原始数据分布')
    plot_kde(synthetic_data[col], color='orange', label='生成数据分布')
    plt.title(f"{col}的原始数据与生成数据分布对比")
    plt.legend()
    plt.show()

# 输出原始数据和合成数据的均值与标准差
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

# 原始数据相关性矩阵
corr_original = train_data.corr()
sns.heatmap(corr_original, annot=False, cmap='coolwarm')
plt.title("原始数据相关性矩阵")
plt.show()

# 合成数据相关性矩阵
corr_synthetic = synthetic_data.corr()
sns.heatmap(corr_synthetic, annot=False, cmap='coolwarm')
plt.title("合成数据相关性矩阵")
plt.show()

# 定义评估函数
def evaluate_model(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(random_state=42,
                                  n_estimators=200,
                                  min_samples_split=2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

# 评估原始数据模型
mse_original, r2_original = evaluate_model(X_train_scaled, y_train, X_test_scaled, y_test)

# 评估增强数据模型
mse_augmented, r2_augmented = evaluate_model(X_train_augmented, y_train_augmented, X_test_scaled, y_test)

# 输出评估结果
print("原始数据模型性能：")
print(f"MSE = {mse_original:.4f}, R² = {r2_original:.4f}")

print("\n数据增强后模型性能: ")
print(f"MSE = {mse_augmented:.4f}, R² = {r2_augmented:.4f}")

# 使用交叉验证评估增强数据
rf = RandomForestRegressor(n_estimators=200, random_state=42)
cv_scores = cross_val_score(rf, X_train_augmented, y_train_augmented, cv=5, scoring='neg_mean_squared_error')
print(f"交叉验证 MSE: {cv_scores.mean()}")
