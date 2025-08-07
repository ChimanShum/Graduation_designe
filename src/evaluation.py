from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sdv.evaluation.single_table import evaluate_quality

def evaluate_model(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, r2, mae

def evaluate_synthetic_data_quality(synthetic_data, train_data, metadata):
    quality_score = evaluate_quality(synthetic_data, train_data, metadata=metadata)
    return quality_score
