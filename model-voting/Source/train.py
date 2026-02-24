
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from pipeline import build_pipelines
from sklearn.ensemble import VotingClassifier
from scipy.stats import uniform, randint

def train_voting_classifier(X, y, numeric_features):
	log_clf, rf_clf, xb_clf = build_pipelines(numeric_features)
	voting_clf = VotingClassifier(
		estimators=[
			('logistic', log_clf),
			('random_forest', rf_clf),
			('xgboost', xb_clf)
		],
		voting='soft'
	)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

	param_dist_random = {
		'logistic__classifier__C': uniform(0.1, 5),
		'logistic__classifier__class_weight': ['balanced', {0: 1, 1: 20}],
		'random_forest__classifier__n_estimators': randint(50, 150),
		'random_forest__classifier__class_weight': ['balanced', 'balanced_subsample'],
		'xgboost__classifier__n_estimators': randint(50, 150),
		'xgboost__classifier__max_depth': randint(3, 7),
		'xgboost__classifier__learning_rate': uniform(0.05, 0.2),
		'xgboost__classifier__scale_pos_weight': [25, 50, 75]
	}
	skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
	random_search = RandomizedSearchCV(
		estimator=voting_clf,
		param_distributions=param_dist_random,
		n_iter=10,
		scoring='recall',
		cv=skf,
		n_jobs=-1,
		random_state=42
	)
	print("Running RandomizedSearchCV for VotingClassifier hyperparameters...")
	random_search.fit(X_train, y_train)
	best_model = random_search.best_estimator_
	best_params = random_search.best_params_
	return best_model, X_test, y_test, best_params
