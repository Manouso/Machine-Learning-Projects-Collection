
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class FeatureEngineer(BaseEstimator, TransformerMixin):
	def __init__(self):
		self.top_features = None
		self.candidate_features = None
	def fit(self, X, y=None):
		if y is not None:
			data_with_y = X.copy()
			data_with_y['Class'] = y
			v_features = [col for col in X.columns if col.startswith('V')]
			class_corr = data_with_y[v_features + ['Class']].corr()['Class'].drop('Class').abs()
			top_class_features = set(class_corr.nlargest(10).index)
			high_error_features = set(['V14', 'V12', 'V10'])
			self.candidate_features = list(top_class_features.intersection(high_error_features))
			if len(self.candidate_features) < 5:
				combined = list(top_class_features.union(high_error_features))
				corr_with_class = class_corr[combined].sort_values(ascending=False)
				self.candidate_features = corr_with_class.head(7).index.tolist()
		else:
			self.candidate_features = ['V14', 'V12', 'V10', 'V17', 'V11', 'V4', 'V16']
		return self
	def transform(self, X):
		X_fe = X.copy()
		X_fe['Amount_Log'] = np.log1p(X_fe['Amount'])
		X_fe['Time_Log'] = np.log1p(X_fe['Time'])
		if self.candidate_features is not None:
			for feature in self.candidate_features[:3]:
				if feature in X_fe.columns:
					X_fe[f'{feature}_squared'] = X_fe[feature] ** 2
			if len(self.candidate_features) >= 2:
				feat1 = self.candidate_features[0]
				feat2 = self.candidate_features[1]
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
