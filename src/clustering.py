import pandas as pd
from tsfeatures import tsfeatures
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import silhouette_score

class TimeSeriesClustering:
    # Топ признаки для кластеризации
    M4_CORE_FEATURES = [
        'trend', # Сила тренда
        'entropy',
        'nonlinearity', # Нелинейность
        'stability', # Стабильность
        'lumpiness', # Всплески
        'hurst' # Память (долгосрочная)
    ]

    @staticmethod
    def extract_features(series_dict, freq=7):
        df_list = []
        for ts_id, series in series_dict.items():
            temp_df = pd.DataFrame({'unique_id': ts_id, 'ds': series.index, 'y': series.values})
            df_list.append(temp_df)
        full_df = pd.concat(df_list, ignore_index=True)

        features = tsfeatures(full_df, freq=freq)
        features = features.fillna(0)
        
        selected_cols = []
        for imp in TimeSeriesClustering.M4_CORE_FEATURES:
            found = [c for c in features.columns if imp in c]
            if found: selected_cols.append(found[0])
            
        # Квантильное преобразование с гауссом - оптимально подобрал исходя из скудной кластеризации
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        scaled_features = qt.fit_transform(features[selected_cols])
        
        return features, scaled_features

    @staticmethod
    def calculate_cluster_metrics(scaled_features, max_k=10):
        """
        Для иерархической кластеризации считаем силуэт.
        """
        k_values = range(2, max_k + 1)
        silhouette_scores = []
        
        for k in k_values:
            # Ward linkage (мин. дисперсия внутри кластеров)
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(scaled_features)
            silhouette_scores.append(silhouette_score(scaled_features, labels))
            
        return k_values, silhouette_scores

    @staticmethod
    def assign_clusters(features_df, scaled_features, n_clusters):
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(scaled_features)
        
        features_copy = features_df.copy()
        features_copy['cluster'] = labels
        
        cluster_dict = dict(zip(features_copy['unique_id'], features_copy['cluster']))
        return cluster_dict, features_copy