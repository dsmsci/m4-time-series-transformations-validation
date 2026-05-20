import pandas as pd
import numpy as np
import warnings

def preprocess_series(series, freq='D'):
    """
    Complete cleaning and preparation of the time series.
    Returns the cleared row and flags for the logs.
    """
    cleaned = series.copy()
    had_nans = False
    had_negatives = False

    # Time sorting
    cleaned = cleaned[~cleaned.index.duplicated(keep='first')]
    cleaned = cleaned.sort_index()
    cleaned = cleaned.asfreq(freq) 

    # NaN Processing
    if cleaned.isna().any():
        had_nans = True
        cleaned = cleaned.interpolate(method='linear')
        cleaned = cleaned.bfill().ffill()

    # Nulls and negatives processing
    min_val = cleaned.min()
    if min_val <= 0:
        had_negatives = True
        shift_amount = abs(min_val) + 1e-3
        cleaned = cleaned + shift_amount

    return cleaned, had_nans, had_negatives


def parse_tsf_file(filepath):
    """
    Parsing .tsf.
    """
    series_data = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        data_started = False
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('@data'):
                data_started = True
                continue
            if not data_started or line.startswith('@'):
                continue
            
            parts = line.split(':')
            ts_id = parts[0]
            values_str = parts[-1].split(',')
            date_str = ":".join(parts[1:-1]).split()[0] 
            
            # Add NaN for symbols
            values = [float(v.strip()) if v.strip() not in ['', '?'] else np.nan for v in values_str]
            series_data.append((ts_id, date_str, values))
            
    return series_data


def get_m4_data(filepath, n_samples=300, start_date='2010-01-01', min_length=150):
    """
    Loading data with a cut-off date of 2010 and a minimum length check.
    """
    print(f"Загрузка и фильтрация данных (Старт: {start_date}, Мин. длина: {min_length})")
    parsed_data = parse_tsf_file(filepath)
    
    df = pd.DataFrame(parsed_data, columns=['ts_id', 'start_date', 'values'])
    
    # Random sampling
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    raw_series_dict = {}
    
    for _, row in df.iterrows():
        ts_id = row['ts_id']
        row_start = pd.to_datetime(row['start_date'])
        
        idx = pd.date_range(start=row_start, periods=len(row['values']), freq='D')
        raw_series = pd.Series(row['values'], index=idx)
        
        truncated_series = raw_series.loc[start_date:]
        
        first_v = truncated_series.first_valid_index()
        last_v = truncated_series.last_valid_index()
        
        if first_v is not None and last_v is not None:
            valid_chunk = truncated_series.loc[first_v:last_v]
            
            if len(valid_chunk) >= min_length:
                raw_series_dict[ts_id] = valid_chunk
                
        if len(raw_series_dict) == n_samples:
            break

    clean_series_dict = {}
    nan_count = 0
    neg_count = 0
    
    for ts_id, series in raw_series_dict.items():
        cleaned, had_nans, had_negatives = preprocess_series(series, freq='D')
        clean_series_dict[ts_id] = cleaned
        if had_nans: nan_count += 1
        if had_negatives: neg_count += 1

    # Logs (in Russian)        
    print(f"\nОтчет по предобработке датасета (отсечение до {start_date}):")
    print(f"Всего рядов собрано: {len(clean_series_dict)}")
    print(f"Рядов с пропусками внутри: {nan_count}")
    print(f"Рядов со сдвигом отрицательных/нулевых значений: {neg_count}")
    print(f"Мин. длина ряда: {min_length} дней")
    
    return clean_series_dict