# FastAPI Credit Card Fraud Prediction Service

A FastAPI-based REST API service that serves predictions from an ensemble voting classifier optimized for detecting fraudulent credit card transactions. This service provides real-time fraud detection capabilities using a pre-trained model from the model-voting project.

## Objectives

- **Real-time Fraud Detection**: Provide instant predictions for credit card transactions via REST API
- **High Recall Focus**: Prioritize catching fraudulent transactions (false negatives are costly)
- **Production-Ready API**: Clean, documented endpoints with proper error handling
- **Model Integration**: Seamlessly integrate with trained ensemble voting models
- **Scalable Architecture**: FastAPI framework for high-performance, asynchronous processing

## Dataset & Model Insights

**Credit Card Fraud Detection Dataset** (Kaggle)
- 284,807 transactions with extreme class imbalance (0.17% fraud rate)
- 30 anonymized features: Time, Amount, and 28 PCA-transformed features (V1-V28)
- Binary classification: Fraud (1) vs. Legitimate (0)

**Model Performance** (Ensemble Voting Classifier):
- **Base Learners**: XGBoost, Random Forest, Logistic Regression
- **Voting Strategy**: Soft voting (probability-based)
- **Recall**: 88.78% (catches 89% of fraudulent transactions)
- **F1-Score**: 69.60% (balance of precision and recall)
- **Precision**: 57.24% (57% of flagged transactions are actually fraud)

**Why High Recall Matters**:
- Missing fraud (false negative) costs money and erodes trust
- Extreme imbalance makes accuracy metrics misleading
- Business priority: Minimize undetected fraud over false alarms

## How It Works

### Architecture
1. **Input Validation**: Pydantic models ensure proper feature format
2. **Model Loading**: Joblib loads the pre-trained voting ensemble model
3. **Preprocessing**: Features converted to DataFrame for model compatibility
4. **Prediction**: Model predicts fraud probability and binary outcome
5. **Response**: Returns prediction results with confidence scores

### API Endpoints

#### POST `/predict`
Predict fraud for a single transaction.

**Request Body** (JSON):
```json
{
  "Time": 10000.0,
  "V1": -1.3598071336738,
  "V2": -0.0727811733098497,
  "V3": 2.53634673796914,
  "V4": 1.37815522427443,
  "V5": -0.338320769942518,
  "V6": 0.462387777762292,
  "V7": 0.239598554061257,
  "V8": 0.0986979012610507,
  "V9": 0.363786969611213,
  "V10": 0.0907941719789316,
  "V11": -0.551599533260813,
  "V12": -0.617800855762348,
  "V13": -0.991389847235408,
  "V14": -0.311169353699879,
  "V15": 1.46817697209427,
  "V16": -0.470400525259478,
  "V17": 0.207971241929242,
  "V18": 0.0257905801985591,
  "V19": 0.403992960255733,
  "V20": 0.251412098239705,
  "V21": -0.018306777944153,
  "V22": 0.277837575558899,
  "V23": -0.110473910188767,
  "V24": 0.0669280749146731,
  "V25": 0.128539358273528,
  "V26": -0.189114843888824,
  "V27": 0.133558376740387,
  "V28": -0.0210530534538215,
  "Amount": 149.62
}
```

**Query Parameters**:
- `model_path` (optional): Path to model file (default: "model-voting/notebooks/models/voting_bayesian_20251207_220624.pkl")

**Response** (JSON):
```json
{
  "prediction": 0,
  "probability": [0.85, 0.15]
}
```
- `prediction`: 0 (legitimate) or 1 (fraud)
- `probability`: Array of probabilities [legitimate_prob, fraud_prob]

## Quick Start

### Prerequisites
- Python 3.8+
- Access to model-voting project (for model file)

### Installation

1. **Clone and navigate to the project**:
   ```bash
   cd fastapi_predict
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model file exists**:
   The service expects the model at `model-voting/notebooks/models/voting_bayesian_20251207_220624.pkl`
   If using a different path, specify via `model_path` query parameter.

### Running the Service

**Development Mode**:
```bash
uvicorn main:app --reload
```

**Production Mode**:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Interactive Documentation

Visit `http://localhost:8000/docs` for Swagger UI documentation with interactive testing.

### cURL Example
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d @sample_transaction.json
```

## Dependencies

- **fastapi**: Modern web framework for building APIs
- **uvicorn**: ASGI server for FastAPI
- **pydantic**: Data validation and serialization
- **joblib**: Model serialization/deserialization
- **pandas**: Data manipulation for model input
- **scikit-learn**: Machine learning utilities

**Production-ready API service for real-time credit card fraud detection using ensemble machine learning models.**
