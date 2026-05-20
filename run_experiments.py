import pandas as pd
import warnings
import os
import joblib
from tqdm import tqdm

from config import *
from src.data import get_m4_data
from src.clustering import TimeSeriesClustering
from src.baselines import get_baselines
from src.validation import WalkForwardValidator

warnings.filterwarnings('ignore')

def main(): # (in Russian)
    series_dict = get_m4_data(filepath=os.path.join(DATA_DIR, 'm4_daily_dataset.tsf'), n_samples=200)

    features_df, scaled_f = TimeSeriesClustering.extract_features(series_dict, freq=SEASONALITY)
    cluster_dict, _ = TimeSeriesClustering.assign_clusters(features_df, scaled_f, N_CLUSTERS)
    
    clusters_data = {i: [(tid, s) for tid, s in series_dict.items() if cluster_dict[tid] == i] for i in range(N_CLUSTERS)}
    
    validator = WalkForwardValidator(HORIZON, N_SPLITS, SEASONALITY, WINDOW_SIZE)
    all_results = []
    models_dir = os.path.join(DATA_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)

    cluster_pbar = tqdm(clusters_data.items(), desc="Кластеры")
    for cluster_id, ts_list in cluster_pbar:
        cluster_pbar.set_description(f"Кластер {cluster_id}")
        
        for transform_name in TRANSFORMS:
            # CatBoost
            cb_results, last_cb = validator.cross_val_global_with_model(cluster_id, ts_list, transform_name)
            all_results.extend(cb_results)

            if last_cb:
                best_l = last_cb.model.get_best_score()['validation']['MultiRMSE']
                tqdm.write(f"Кластер {cluster_id} с {transform_name}. CB: {last_cb.model.tree_count_} деревьев, Лосс: {best_l:.4f}")
                joblib.dump(last_cb, os.path.join(models_dir, f'cb_c{cluster_id}_{transform_name}.joblib'))
            
            # StatsForecast
            sf_models = get_baselines(SEASONALITY)
            sf_res = validator.cross_val_local_statsforecast(cluster_id, ts_list, sf_models, transform_name)
            all_results.extend(sf_res)
            tqdm.write(f"Кластер {cluster_id} с {transform_name}. Stats: готово")

    # Final saving
    pd.DataFrame(all_results).to_csv(os.path.join(RESULTS_DIR, 'metrics.csv'), index=False)
    print("\nМетрики в metrics.csv")

if __name__ == '__main__':
    main()