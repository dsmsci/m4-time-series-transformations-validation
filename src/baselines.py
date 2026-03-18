from statsforecast.models import Naive, SeasonalNaive, AutoTheta, AutoETS, AutoARIMA

def get_baselines(seasonality):
    return [
        Naive(),
        SeasonalNaive(season_length=seasonality),
        AutoTheta(season_length=seasonality),
        AutoETS(season_length=seasonality),
        AutoARIMA(season_length=seasonality, stepwise=True)
    ]