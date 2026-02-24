


import pandas as pd
from preprocessing import feature_engineering, get_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import FunctionTransformer

def build_pipelines(numeric_features):
	preprocessor = get_preprocessor(numeric_features)
	log_clf = Pipeline([
		('feature_engineering', FunctionTransformer(feature_engineering)),
		('preprocessor', preprocessor),
		('classifier', LogisticRegression(solver='liblinear', max_iter=1000))
	])
	rf_clf = Pipeline([
		('feature_engineering', FunctionTransformer(feature_engineering)),
		('preprocessor', preprocessor),
		('classifier', RandomForestClassifier(random_state=42))
	])
	xb_clf = Pipeline([
		('feature_engineering', FunctionTransformer(feature_engineering)),
		('preprocessor', preprocessor),
		('classifier', xgb.XGBClassifier(random_state=42, n_jobs=1))
	])
	return log_clf, rf_clf, xb_clf
