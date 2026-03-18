import numpy as np
from scipy.stats import boxcox, boxcox_normmax
from scipy.special import inv_boxcox

class TimeSeriesTransformer:
    def __init__(self, method='none'):
        self.method = method
        self.lambda_ = None
        self.mean = 0.0
        self.std = 1.0
        self.last_val = None
        
        self.val_min = None
        self.val_max = None
        self.is_constant = False # флагируем константный ряд (вдруг такой есть, а че бы и нет)

    def fit(self, series):
        series = np.array(series, dtype=float)

        if np.any(series <= 0):
            raise ValueError("Box-Cox требует строго положительных значений") # с М4 такого нет + мы провели предочистку
        
        # Если все значения равны первому, ряд константный
        if np.allclose(series, series[0]):
            self.is_constant = True
            self.last_val = series[0]
            return self
            
        if self.method == 'boxcox':
            try:
                self.lambda_ = boxcox_normmax(series, method='mle')
            except Exception:
                self.lambda_ = 0.0

            transformed = boxcox(series, lmbda=self.lambda_)
            
            # Параметры скейлинга
            self.mean = np.mean(transformed)
            self.std = np.std(transformed)
            if self.std < 1e-9: 
                self.std = 1.0
                
            # Робастный расчет области допустимых через квантили (защита от выбросов в трейне)
            q01 = np.quantile(transformed, 0.01)
            q99 = np.quantile(transformed, 0.99)
            margin = (q99 - q01) * 0.5
            
            self.val_min = q01 - margin
            self.val_max = q99 + margin
            
        return self

    def transform(self, series):
        series = np.array(series, dtype=float)
        self.last_val = series[-1]
        
        # Если ряд константный, нет смысла его трансформировать
        if self.is_constant or self.method == 'none':
            return series
            
        if self.method == 'log1p':
            return np.log1p(series)
        elif self.method == 'boxcox':
            transformed = boxcox(series, lmbda=self.lambda_)
            return (transformed - self.mean) / self.std
        elif self.method == 'diff':
            return np.diff(series, prepend=series[0])

    def fit_transform(self, series):
        return self.fit(series).transform(series)
            
    def inverse_transform(self, preds):
        preds = np.array(preds, dtype=float)
        
        # Если ряд был константным, жеско возвращаем эту константу
        if self.is_constant:
            return np.full_like(preds, self.last_val)
            
        if self.method == 'none':
            return preds
        elif self.method == 'log1p':
            return np.expm1(preds)
        elif self.method == 'boxcox':
            val_unscaled = preds * self.std + self.mean
            
            # Робастный клиппинг по границам квантилей
            val_clipped = np.clip(val_unscaled, self.val_min, self.val_max)
            
            # Математическая граница Бокса-Кокса (защита от получения отрицательных значений под корнем)
            eps = 1e-8
            if abs(self.lambda_) > eps:
                boundary = -1.0 / self.lambda_
                if self.lambda_ > 0:
                    val_clipped = np.maximum(val_clipped, boundary + eps)
                else:
                    val_clipped = np.minimum(val_clipped, boundary - eps)
            
            # Инверсия
            res = inv_boxcox(val_clipped, self.lambda_)
            
            # Последний рубеж защиты от NaN (если совсем плохо получится - возвращаем наивный прогноз)
            return np.nan_to_num(res, nan=self.last_val, posinf=self.last_val, neginf=self.last_val)
            
        elif self.method == 'diff':
            return self.last_val + np.cumsum(preds)