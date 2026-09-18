from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model(alpha, seed=42):
    return Pipeline([('scaler', StandardScaler()), ('regressor', Ridge(alpha=alpha, random_state=seed))])
