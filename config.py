import os

# Папки выделены таким образом, чтобы учесть, что репозиторий может запускаться на разных ОС (тип путей к директориям разный)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# На основе EDA выделены следующие параметры эксперимента:
HORIZON = 14 # Горизонт прогнозирования
WINDOW_SIZE = 30 # Размер исторического окна для CatBoost
SEASONALITY = 7 # Сезонность

# Параметры моделей на основе EDA и гипотезы:
N_CLUSTERS = 3 # Количество кластеров
N_SPLITS = 3 # Количество фолдов
TRANSFORMS = ['none', 'log1p', 'boxcox', 'diff']

# Отдельные гиперпараметры для CatBoost
CB_ITERATIONS = 1000 # Максимальное число деревьев
CB_EARLY_STOPPING = 50 # Early stopping для CatBoost - количество шагов для того, чтобы произошла остановка
CB_VAL_RATIO = 0.1 # Доля последних окон для валидационного окна 