import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

class GlobalCatBoostMIMO:
    def __init__(self, horizon, window_size, iterations=1000, early_stopping_rounds=50):
        self.horizon = horizon
        self.window_size = window_size
        self.iterations = iterations
        self.early_stopping_rounds = early_stopping_rounds
        # Fixed column names (CB will be able to evaluate features better this way)
        self.col_names = [f"lag_{i}" for i in range(self.window_size)] + ["ts_id"]
        
        self.model = CatBoostRegressor(
            iterations=self.iterations, 
            loss_function='MultiRMSE', 
            early_stopping_rounds=self.early_stopping_rounds,
            random_seed=42,
            verbose=0
        )
        
    def _extract_windows(self, values, ts_id):
        X, Y = [], []
        for i in range(len(values) - self.window_size - self.horizon + 1):
            window = values[i : i + self.window_size]
            target = values[i + self.window_size : i + self.window_size + self.horizon]
            X.append(list(window) + [str(ts_id)])
            Y.append(target)
        return X, Y

    def fit(self, train_data_list, val_ratio=0.1):
        X_train, Y_train = [], []
        X_val, Y_val = [], []
        
        for ts_id, values in train_data_list:
            X_s, Y_s = self._extract_windows(values, ts_id)
            if len(X_s) < 2: # Minimum 2 windows for split
                continue 
                
            split = int(len(X_s) * (1 - val_ratio))
            # At least one window will be validated
            split = min(max(split, 1), len(X_s) - 1)
            
            X_train.extend(X_s[:split])
            Y_train.extend(Y_s[:split])
            X_val.extend(X_s[split:])
            Y_val.extend(Y_s[split:])

        if len(X_train) == 0:
            # (in Russian)
            raise ValueError("Недостаточно данных для формирования окон.")

        if len(X_val) > 0:
            self.model.fit(
                pd.DataFrame(X_train, columns=self.col_names), np.array(Y_train),
                eval_set=(pd.DataFrame(X_val, columns=self.col_names), np.array(Y_val)),
                cat_features=['ts_id'],
                use_best_model=True
            )
        else:
            self.model.fit(
                pd.DataFrame(X_train, columns=self.col_names), np.array(Y_train),
                cat_features=['ts_id']
            )
        
    def predict(self, test_data_list):
        X_test = []
        for ts_id, values in test_data_list:
            # Padding: instead of an error, we stretch the row to the required length - not used (in validation.py, such rows are filtered, just as an alternative)
            if len(values) < self.window_size:
                pad_size = self.window_size - len(values)
                window = np.pad(values, (pad_size, 0), mode='edge')
            else:
                window = values[-self.window_size:]
            
            X_test.append(list(window) + [str(ts_id)])
            
        return self.model.predict(pd.DataFrame(X_test, columns=self.col_names))