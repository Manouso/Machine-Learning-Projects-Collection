
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

def evaluate_model(model, X_test, y_test):
	y_pred = model.predict(X_test)
	y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
	report_dict = classification_report(y_test, y_pred, output_dict=True)
	cm = confusion_matrix(y_test, y_pred)
	acc = accuracy_score(y_test, y_pred)
	roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
	class1 = report_dict['1']
	return {
		'recall': class1['recall'],
		'precision': class1['precision'],
		'f1-score': class1['f1-score'],
		'support': class1['support'],
		'classification_report': classification_report(y_test, y_pred),
		'confusion_matrix': cm,
		'accuracy': acc,
		'roc_auc': roc_auc
	}
