import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import random
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 设置字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei'] # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 读取数据并删除缺失值
df = pd.read_excel('data_without_some_cols.xlsx', sheet_name='Sheet1').dropna()
# 分离特征和目标变量
X = df.drop("CO2 uptake", axis=1)
y = df["CO2 uptake"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化特征（树模型不需要，但线性模型需要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义模型列表
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42,
                                           max_depth=10,
                                           min_samples_split=2,),
    "Random Forest": RandomForestRegressor(random_state=42,
                                        #    max_depth=10,
                                           min_samples_split=2,
                                           n_estimators=200),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42,
                                           max_depth=10,
                                           min_samples_split=2,
                                           n_estimators=200),
    "Support Vector Machine": SVR()
}

# 评估函数
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {
        "MSE": round(mse, 4),
        "MAE": round(mae, 4),
        "R²": round(r2, 4)
    }

# 存储结果
results = {}

# 遍历所有模型
for name, model in models.items():
    if "Linear" in name or "SVM" in name:
        # 线性模型使用标准化后的数据
        result = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
    else:
        # 树模型使用原始数据
        result = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[name] = result

# 输出结果
print("模型性能对比：")
for model, metrics in results.items():
    print(f"{model}:")
    print(f"  MSE = {metrics['MSE']}, MAE = {metrics['MAE']}, R² = {metrics['R²']}")


# 选择性能最好的模型（例如随机森林）
best_model = RandomForestRegressor(random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# 绘制对比图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("实际值")
plt.ylabel("预测值")
plt.title("实际值 vs 预测值（随机森林）")
plt.show()

# 随机森林的特征重要性
feature_importance = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar')
plt.title("特征重要性（随机森林）")
plt.ylabel("重要性")
plt.xticks(rotation=45)
plt.show()

from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}

# 网格搜索
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)
grid_search.fit(X_train, y_train)

# 输出最优参数
print("最优参数:", grid_search.best_params_)
optimized_gb = grid_search.best_estimator_
y_pred = optimized_gb.predict(X_test)
print("最优模型MSE:", round(-grid_search.best_score_, 4))
print("优化后R²:", r2_score(y_test, y_pred))