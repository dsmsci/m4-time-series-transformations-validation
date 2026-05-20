import numpy as np

def smape(y_true, y_pred, epsilon=1e-6):
    """
    Modified sMAPE.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    sum_abs = np.abs(y_true) + np.abs(y_pred)
    abs_max = np.maximum(sum_abs + epsilon, 0.5 + epsilon) / 2.0
    
    diff = np.abs(y_true - y_pred) / abs_max
    return np.mean(diff) * 100

def mase(y_true, y_pred, y_train, seasonality=1):
    """
    MASE with Naive Seasonal forecast.
    """
    forecast_error = np.mean(np.abs(y_true - y_pred))
    if len(y_train) > seasonality:
        naive_error = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    else:
        # If the series are too short, uses lag 1
        naive_error = np.mean(np.abs(y_train[1:] - y_train[:-1]))
        
    if naive_error < 1e-6: # If the naive error is too small -> np.nan
        return np.nan
        
    return forecast_error / naive_error

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

def calculate_all_metrics(y_true, y_pred, y_train, seasonality):
    return {
        'sMAPE': smape(y_true, y_pred),
        'MASE': mase(y_true, y_pred, y_train, seasonality),
        'RMSE': rmse(y_true, y_pred)
    }