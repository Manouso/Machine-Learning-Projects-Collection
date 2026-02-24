
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def feature_engineering(X):
    X_fe = X.copy()
    # Log transformations
    X_fe['Amount_Log'] = np.log1p(X_fe['Amount'])
    X_fe['Time_Log'] = np.log1p(X_fe['Time'])
    # Add polynomial and interaction features for selected columns
    candidate_features = ['V14', 'V12', 'V10', 'V17', 'V11', 'V4', 'V16']

    for feature in candidate_features[:3]:
        if feature in X_fe.columns:
            X_fe[f'{feature}_squared'] = X_fe[feature] ** 2
    if len(candidate_features) >= 2:
        feat1 = candidate_features[0]
        feat2 = candidate_features[1]
        if feat1 in X_fe.columns and feat2 in X_fe.columns:
            X_fe[f'{feat1}_x_{feat2}'] = X_fe[feat1] * X_fe[feat2]
    return X_fe

def get_preprocessor(numeric_features):
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features)
    ])
    return preprocessor
