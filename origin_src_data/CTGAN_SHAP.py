import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sdv.single_table import CTGANSynthesizer  # 使用CTGAN生成表格数据
from sdv.metadata import SingleTableMetadata
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import gaussian_kde
from sdv.evaluation.single_table import evaluate_quality
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
import shap  # 导入shap库

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 设置字体
plt.rcParams["font.sans-serif"] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取原始数据
df = pd.read_excel('data_without_some_cols.xlsx', sheet_name='Sheet1').dropna()
high_variability_columns = ['SSABD', 'SSAAD', 'MW', 'PPM', 'LA']
df[high_variability_columns] = df[high_variability_columns].apply(lambda x: np.log(x + 1))
X = df.drop("CO2 uptake", axis=1)
y = df["CO2 uptake"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 合并训练集的特征和目标变量
train_data = pd.concat([X_train, y_train], axis=1)

# 自动推断数据元数据
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_data)

# 初始化CTGAN模型（专为表格数据设计）
model = CTGANSynthesizer(epochs=1000,
                         verbose=True,
                         metadata=metadata,
                         )
model.fit(train_data)

# 生成与原始训练集相同数量的合成数据
synthetic_data = model.sample(len(train_data))

# 分离合成数据的特征和目标变量
X_syn = synthetic_data.drop("CO2 uptake", axis=1)
y_syn = synthetic_data["CO2 uptake"]

# 合并原始训练集和合成数据
X_train_augmented = pd.concat([X_train, X_syn], axis=0)
y_train_augmented = pd.concat([y_train, y_syn], axis=0)

# 随机森林模型训练
rf_model = RandomForestRegressor(random_state=42,
                                  n_estimators=200,
                                  min_samples_split=2,
                                  # max_depth=10
                                  )
rf_model.fit(X_train, y_train)

def evaluate_model(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(random_state=42,
                                  n_estimators=200,
                                  min_samples_split=2,
                                  # max_depth=10
                                  )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, r2, mae

# 评估原始数据模型
mse_original, r2_original, mae_original = evaluate_model(X_train, y_train, X_test, y_test)

# SHAP分析
explainer = shap.TreeExplainer(rf_model)  # 使用树模型的解释器
shap_values = explainer.shap_values(X_test)

# 可视化SHAP值
shap.summary_plot(shap_values, X_test)

# 绘制SHAP值的特征重要性
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar")
plt.title('平均SHAP值')  # 设置直方图标题
plt.show()

# 输出结果
print("原始数据模型性能：")
print(f"MSE = {mse_original:.4f}, R² = {r2_original:.4f}, MAE = {mae_original:.4f}")

# 生成数据与真实数据的相似得分
print(f'生成数据与真实数据的相似得分：{evaluate_quality(synthetic_data, train_data, metadata=metadata)}')
