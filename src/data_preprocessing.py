import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def load_and_preprocess_raw_data(file_path):
    """
    读取原始数据并进行预处理
    """
    data = pd.read_excel(file_path).dropna()
    
    # 进行数值列的正态性检验
    normality_results = check_normality(data)
    print("正态性检验结果:", normality_results)

    # 计算相关性矩阵
    corr_matrix = calculate_correlation(data)
    print("相关性矩阵:", corr_matrix)

    # 去除高相关性特征
    data_cleaned, dropped_cols = drop_high_correlation_columns(data)
    print(f"去除的列：{dropped_cols}")
    
    # 计算VIF
    vif_data = calculate_vif(data_cleaned)
    print("VIF结果:", vif_data)

    # 处理后的数据保存
    data_cleaned.to_excel('data_without_some_cols.xlsx', index=False)

    return data_cleaned

def check_normality(data):
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    normality_results = {col: stats.shapiro(data[col].dropna()) for col in numeric_columns}
    return normality_results

def calculate_correlation(data):
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = data[numeric_columns].corr(method='spearman')
    return corr_matrix

def drop_high_correlation_columns(data, correlation_threshold=0.7):
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = data[numeric_columns].corr(method='spearman')
    high_correlation_pairs = [
        (col1, col2) for col1 in corr_matrix.columns for col2 in corr_matrix.columns
        if col1 != col2 and abs(corr_matrix[col1][col2]) >= correlation_threshold
    ]
    cols_to_drop = {col1 for col1, _ in high_correlation_pairs}
    data_cleaned = data.drop(columns=cols_to_drop)
    return data_cleaned, cols_to_drop

def calculate_vif(data):
    """
    计算方差膨胀因子（VIF）来评估多重共线性
    """
    # 增加常数项
    X = data
    X_const = add_constant(X)

    # 计算VIF
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

    # 输出VIF结果
    return vif_data.sort_values(by="VIF", ascending=False)
