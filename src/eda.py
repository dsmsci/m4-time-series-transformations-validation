import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf

class TimeSeriesEDA:
    @staticmethod
    def check_stationarity(series, signif=0.05):
        """
        ADF and KPSS Stationarity Test.
        Returns (is_adf_stationary, is_kpss_stationary).
        """
        try:
            adf_res = adfuller(series.dropna(), autolag='AIC')
            is_adf_stat = adf_res[1] < signif
        except ValueError:
            is_adf_stat = False

        try:
            # Silencing KPSS warnings about interpolation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_res = kpss(series.dropna(), regression='c', nlags="auto")
            is_kpss_stat = kpss_res[1] >= signif
        except ValueError:
            is_kpss_stat = False

        return is_adf_stat, is_kpss_stat

    @staticmethod
    def extract_stl_features(series, period):
        """
        STL decomp.
        """
        stl = STL(series, period=period, robust=True).fit()
        var_resid = np.var(stl.resid)
        var_trend = np.var(stl.trend + stl.resid)
        var_season = np.var(stl.seasonal + stl.resid)
        
        # Normalized strength of trend and seasonality
        strength_trend = max(0, 1 - var_resid / var_trend) if var_trend > 0 else 0
        strength_season = max(0, 1 - var_resid / var_season) if var_season > 0 else 0
        
        return strength_trend, strength_season

    @staticmethod
    def plot_stl_multi(ts_ids, series_dict, period):
        """
        STL plots (in Russian).
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Исходные ряды", "Тренды (STL)", "Сезонность (STL)", "Остатки (STL)"),
            horizontal_spacing=0.07, vertical_spacing=0.12
        )
        for ts_id in ts_ids:
            series = series_dict[ts_id].dropna()
            stl = STL(series, period=period).fit()
            fig.add_trace(go.Scatter(x=series.index, y=series, name=f"{ts_id}"), row=1, col=1)
            fig.add_trace(go.Scatter(x=series.index, y=stl.trend, name=f"{ts_id} Trend"), row=1, col=2)
            fig.add_trace(go.Scatter(x=series.index, y=stl.seasonal, name=f"{ts_id} Season"), row=2, col=1)
            fig.add_trace(go.Scatter(x=series.index, y=stl.resid, mode='markers', name=f"{ts_id} Resid", marker=dict(size=4)), row=2, col=2)
        
        fig.update_layout(height=700, width=1100, template="plotly_white", title="STL Анализ")
        return fig

    @staticmethod
    def plot_corr_multi(ts_ids, series_dict, lags=40):
        """
        ACF, PACF plots (in Russian)
        """
        fig = make_subplots(
            rows=1, cols=2, 
            subplot_titles=(f"ACF (lags={lags})", f"PACF (lags={lags})"),
            horizontal_spacing=0.07
        )
        lags_idx = np.arange(lags + 1)
        for ts_id in ts_ids:
            series = series_dict[ts_id].dropna()
            acf_vals = acf(series, nlags=lags)
            pacf_vals = pacf(series, nlags=lags, method='ywm')
            
            fig.add_trace(go.Scatter(x=lags_idx, y=acf_vals, name=f"{ts_id} ACF", mode='lines+markers'), row=1, col=1)
            fig.add_trace(go.Scatter(x=lags_idx, y=pacf_vals, name=f"{ts_id} PACF", mode='lines+markers'), row=1, col=2)
            
        fig.add_hline(y=0, line_color="black", opacity=0.3)
        fig.update_layout(height=400, width=1100, template="plotly_white", title="Корреляционный анализ")
        return fig
    
    @classmethod
    def analyze_dataset(cls, series_dict, period):
        """
        Aggregation of analysis across all series.
        """
        results = []
        for ts_id, series in series_dict.items():
            adf_stat, kpss_stat = cls.check_stationarity(series)
            try:
                trend_str, season_str = cls.extract_stl_features(series.dropna(), period)
            except ValueError:
                trend_str, season_str = 0.0, 0.0

            results.append({
                'ts_id': ts_id, 
                'adf_stationary': adf_stat,
                'kpss_stationary': kpss_stat,
                'trend_strength': trend_str, 
                'seasonality_strength': season_str
            })
            
        return pd.DataFrame(results)