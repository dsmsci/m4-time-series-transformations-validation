import pandas as pd
from statsforecast import StatsForecast
from src.metrics import calculate_all_metrics
from src.transforms import TimeSeriesTransformer
from src.models import GlobalCatBoostMIMO
from config import CB_ITERATIONS, CB_EARLY_STOPPING, CB_VAL_RATIO
from tqdm import tqdm

class WalkForwardValidator:
    def __init__(self, horizon, n_splits, seasonality, window_size=30):
        self.horizon = horizon
        self.n_splits = n_splits
        self.seasonality = seasonality
        self.window_size = window_size

    def _get_fold_indices(self, series_length, fold):
        """
        Returns the train and test indices from the end of the series.
        """
        steps_from_end = self.n_splits - fold
        test_start = series_length - steps_from_end * self.horizon
        test_end = test_start + self.horizon
        
        if test_start <= self.window_size: 
            return None, None
            
        train_idx = list(range(0, test_start))
        test_idx = list(range(test_start, test_end))
        return train_idx, test_idx
    
    def cross_val_local_statsforecast(self, cluster_id, ts_list, models, transform_name):
        all_fold_results = []
        
        for fold in tqdm(range(self.n_splits), desc=f"Фолды Stats", leave=False): # at the early stage of the experiment
            train_dfs = []
            test_data_map = {}
            transformers_map = {}

            for ts_id, series in ts_list:
                train_idx, test_idx = self._get_fold_indices(len(series), fold)
                if train_idx is None:
                    continue

                train_ser = series.iloc[train_idx]
                test_ser = series.iloc[test_idx]

                split_idx = int(len(train_ser) * (1 - CB_VAL_RATIO))
                transformer = TimeSeriesTransformer(transform_name)
                transformer.fit(train_ser.values[:split_idx])
                
                transformed_y = transformer.transform(train_ser.values)
                
                train_dfs.append(pd.DataFrame({
                    'unique_id': [ts_id] * len(transformed_y),
                    'ds': train_ser.index,
                    'y': transformed_y
                }))
                
                test_data_map[ts_id] = test_ser.values
                transformers_map[ts_id] = transformer

            if not train_dfs:
                continue

            df_long = pd.concat(train_dfs, ignore_index=True)

            # Launches models for the long data format
            sf = StatsForecast(models=models, freq='D', n_jobs=-1)
            
            sf.fit(df=df_long)
            
            forecasts_df = sf.predict(h=self.horizon).reset_index()

            # Extracting parameters of car models
            fitted_models = sf.fitted_ 
            uids = sf.uids 
            
            params_map = {}
            for row_idx, uid in enumerate(uids):
                params_map[uid] = {}
                for col_idx, model_obj in enumerate(fitted_models[row_idx]):
                    m_name = type(models[col_idx]).__name__
                    try:
                        if hasattr(model_obj, 'model_'):
                            m_info = model_obj.model_
                            if isinstance(m_info, dict):
                                if m_name == 'AutoETS':
                                    # ETS(A,N,N) string
                                    p_str = m_info.get('method', 'ETS') 
                                elif m_name == 'AutoARIMA':
                                    # (p,d,q,P,D,Q,m) string
                                    arma = m_info.get('arma', '')
                                    p_str = f"ARIMA{arma}" if arma else "ARIMA"
                                elif m_name == 'AutoTheta':
                                    # Model type string
                                    p_str = m_info.get('modeltype', 'Theta')
                                else:
                                    p_str = "fitted"
                            else:
                                p_str = str(m_info)
                        else:
                            p_str = str(model_obj)
                    except:
                        p_str = "fitted"
                        
                    params_map[uid][m_name] = p_str

            # Inverse transformation and saving the model
            model_names = [type(m).__name__ for m in models]
            for ts_id in forecasts_df['unique_id'].unique():
                ts_forecasts = forecasts_df[forecasts_df['unique_id'] == ts_id]
                transformer = transformers_map[ts_id]
                y_true = test_data_map[ts_id]

                for m_name in model_names:
                    preds_tf = ts_forecasts[m_name].values
                    preds_orig = transformer.inverse_transform(preds_tf)
                    
                    # MASE requires a source train
                    y_train_orig = next(s for i, s in ts_list if i == ts_id).iloc[:len(y_true)*-1].values
                    
                    metrics = calculate_all_metrics(y_true, preds_orig, y_train_orig, self.seasonality)
                    
                    all_fold_results.append({
                        'ts_id': ts_id, 'cluster': cluster_id, 'transform': transform_name,
                        'model': m_name, 'fold': fold, 
                        'auto_params': params_map[ts_id].get(m_name, ""),
                        **metrics
                    })
                    
        return all_fold_results
    
    def cross_val_global_with_model(self, cluster_id, ts_list, transform_name):
        """
        Validation of global CatBoost.
        """
        fold_results = []
        last_trained_model = None
        
        for fold in tqdm(range(self.n_splits), desc=f"Фолды CatBoost", leave=False):
            cb_train_transformed, transformers, y_true_dict, y_train_dict, valid_ts_list = [], {}, {}, {}, []
            
            for ts_id, series in ts_list:
                train_idx, test_idx = self._get_fold_indices(len(series), fold)
                if train_idx is None:
                    continue 
                    
                train, test = series.iloc[train_idx], series.iloc[test_idx]
                y_true_dict[ts_id], y_train_dict[ts_id] = test.values, train.values
                
                split_idx = int(len(train) * (1 - CB_VAL_RATIO))
                transformer = TimeSeriesTransformer(transform_name)
                transformer.fit(train.values[:split_idx])
                
                # Add a tuple (ts_id, transformed_values)
                transformed_vals = transformer.transform(train.values)
                cb_train_transformed.append((ts_id, transformed_vals))
                
                transformers[ts_id] = transformer
                valid_ts_list.append((ts_id, series))

            if not cb_train_transformed:
                continue

            cb_model = GlobalCatBoostMIMO(
                self.horizon, self.window_size, 
                iterations=CB_ITERATIONS, early_stopping_rounds=CB_EARLY_STOPPING
            )
            
            try:
                cb_model.fit(cb_train_transformed, val_ratio=CB_VAL_RATIO)
                cb_preds_tf = cb_model.predict(cb_train_transformed)
                
                for idx, (ts_id, _) in enumerate(valid_ts_list):
                    preds = transformers[ts_id].inverse_transform(cb_preds_tf[idx])
                    metrics = calculate_all_metrics(y_true_dict[ts_id], preds, y_train_dict[ts_id], self.seasonality)
                    
                    fold_results.append({
                        'ts_id': ts_id, 'cluster': cluster_id, 'transform': transform_name,
                        'model': 'CatBoost_MIMO', 'fold': fold, 'auto_params': '', **metrics 
                    })
                last_trained_model = cb_model
            except Exception as e:
                print(f"Ошибка CatBoost: {e}")
                
        return fold_results, last_trained_model