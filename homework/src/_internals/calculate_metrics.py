# Metricas de error durante testing

from homework.train_knn import estimator, x_test, y_test
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(modelo, x, y):
    y_pred = estimator.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse,mae,r2