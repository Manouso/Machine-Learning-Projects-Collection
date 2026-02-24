'''Examples:
    python predict.py --model-path model-voting/models/voting_bayesian_YYYYMMDD_HHMMSS.pkl \
        --features-json model-voting/sample_feature.json --print-proba

Notes:
- The saved artifacts in `model-voting` are pipelines (preprocessing + model), so pass raw feature values
    with the same column names as the training data.

'''

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

# file handling
import argparse
import json
import os
import sys
from typing import Any, Dict

# data handling
import numpy as np
import pandas as pd


# Use joblib (models in model-voting are saved via joblib)
import joblib


# --- Function-based feature engineering ---
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

# Load a saved model or pipeline
def load_model(model_path: str) -> Any:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

# Parse command-line arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict using a saved model/pipeline (model-voting)")
    parser.add_argument('--model-path', type=str, required=True) # Path to the saved model or pipeline file (.pkl)
    parser.add_argument('--features-json', type=str, required=True) # Path to a JSON file containing a single object with feature:value pairs (matching training columns)
    parser.add_argument('--output', type=str, default='') # Optional path to write JSON result
    return parser.parse_args()

# Build input DataFrame from JSON file
def load_features(features_json_path: str) -> pd.DataFrame:
    """
    Load features from a JSON file. Supports either a single dict or a list of dicts.
    Returns a DataFrame with one row per example.
    """
    if not os.path.exists(features_json_path):
        raise FileNotFoundError(f"Features JSON not found: {features_json_path}")
    with open(features_json_path, 'r') as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return pd.DataFrame([obj])
    elif isinstance(obj, list):
        if not all(isinstance(row, dict) for row in obj):
            raise ValueError("All items in the features list must be dictionaries.")
        return pd.DataFrame(obj)
    else:
        raise ValueError("Features JSON must be a dict or a list of dicts.")

# Predict for a batch of samples
def predict_batch(model, X: pd.DataFrame):
    """
    Predict for a batch of samples. Returns a list of predictions.
    """
    return model.predict(X)

if __name__ == '__main__':
    args = parse_args()
    try:
        model = load_model(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        X = load_features(args.features_json)
    except Exception as e:
        print(f"Error loading features from {args.features_json}: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        preds = predict_batch(model, X)
        for i, pred in enumerate(preds):
            print(f"Sample {i}: Predicted label: {pred}")
    except Exception as e:
        print("Error during prediction. This usually means the input feature names or ordering don't match the saved model.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        print("Input columns:", list(X.columns), file=sys.stderr)
        sys.exit(3)

    # Optional JSON output
    if args.output:
        try:
            output = []
            for i in range(len(X)):
                output.append({'input': X.iloc[i].to_dict(), 'prediction': int(preds[i])})
            with open(args.output, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Saved JSON output to {args.output}")
        except Exception as e:
            print(f"Failed to write output JSON: {e}", file=sys.stderr)