from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model(params, seed=42):
    return Pipeline([('scaler', StandardScaler()), ('regressor', RandomForestRegressor(
        criterion='squared_error', random_state=seed, **params))])
