from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from evaluation import evaluate_synthetic_data_quality


def train_model(model, X_train, y_train, X_test, y_test):
    """
    训练模型并评估性能
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, r2, mae

def evaluate_all_models(X_train, X_test, y_train, y_test):
    """
    对所有模型进行训练和评估
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10, min_samples_split=2),
        "Random Forest": RandomForestRegressor(random_state=42, min_samples_split=2, n_estimators=200),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42, max_depth=10, min_samples_split=2, n_estimators=200),
        "Support Vector Machine": SVR()
    }
    
    results = {}
    best_model_name = None
    best_model = None
    best_performance = float('inf')  # 初始为正无穷大，用来找最好的模型

    for name, model in models.items():
        if "Linear" in name or "SVM" in name:
            # 线性模型使用标准化后的数据
            result = train_model(model, X_train, y_train, X_test, y_test)
        else:
            # 树模型使用原始数据
            result = train_model(model, X_train, y_train, X_test, y_test)
        
        mse, r2, mae = result
        results[name] = result
        
        # 选择表现最好的模型（MSE最小）
        if mse < best_performance:
            best_performance = mse
            best_model_name = name
            best_model = model

    return results, best_model_name, best_model

def combine_model_and_synthetic_data(X_train, y_train, X_test, y_test, synthetic_data, best_model):
    """
    将生成的数据与最佳回归模型（如随机森林）结合，并评估性能
    """
    # 评估合成数据的质量
    print("\n评估合成数据与真实数据的相似性...")
    synthetic_metrics = evaluate_synthetic_data_quality(synthetic_data, X_train, metadata=None)  # 无需metadata
    print(f"生成数据与真实数据相似得分: {synthetic_metrics}")

    # 评估生成数据与最佳模型结合后的性能
    print("\n评估生成数据与最佳模型的性能...")
    mse, r2, mae = train_model(best_model, X_train, y_train, X_test, y_test)  # 使用最佳模型

    print("\n最终模型性能：")
    print(f"MSE = {mse:.4f}, R² = {r2:.4f}, MAE = {mae:.4f}")
