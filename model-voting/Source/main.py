
import pandas as pd
from train import train_voting_classifier
from evaluate import evaluate_model

# main entry point for ML workflow
def main():
    # Load data
    data = pd.read_csv('../datasets/creditcard.csv')
    X = data.drop('Class', axis=1)
    y = data['Class']
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Train model with hyperparameter tuning
    voting_clf, X_test, y_test, best_params = train_voting_classifier(X, y, numeric_features)
    
    # Print best hyperparameters
    print('Best Hyperparameters:', best_params)

    # Evaluate
    results = evaluate_model(voting_clf, X_test, y_test)

    print('Results:')
    print(results['classification_report'])
    print(results['confusion_matrix'])

    print('--- Class 1 (Fraud) Metrics ---')
    print(f"Recall:    {results['recall']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"F1-score:  {results['f1-score']:.4f}")
    print(f"Support:   {int(results['support'])}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"ROC AUC:   {results['roc_auc']:.4f}")

if __name__ == '__main__':
    main()
