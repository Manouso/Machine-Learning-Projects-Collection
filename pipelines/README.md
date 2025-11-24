# Machine Learning Pipelines Projects

This repository contains two machine learning projects demonstrating how to use **scikit-learn Pipelines** for real-world preprocessing, model training, hyperparameter tuning, and evaluation.

Pipelines are essential in modern ML workflows because they:
- Prevent **data leakage**
- Automatically apply preprocessing inside cross-validation
- Keep code clean and modular
- Make hyperparameter tuning safer and easier
- Ensure reproducible transformations
- Simplify deployment

---

## Project Structure

- **pipeline/**
  
  - `pipeline_logistic_regression.ipynb` — notebook for titanic dataset
  - `pipeline_logistic_regression.ipynb` — notebook for housing dataset
  - `requirements.txt` — Requirements  
  - `README.md` — Project overview

---

## Objectives

- Understand why pipelines are useful
- Build pipeline that includes:
  - Feature Engineering
  - Eliminate Features
  - Numeric Preprocessing
  - Categorical Preprocessing
  - Logistic Regression Classifier
- Perform **Hyperparameter Tuning** within the pipeline RandomSearchCV
- Evaluate and compare:
  - Accuracy with vs. without pipeline
  - Execution time
  - Feature Importance  
  
---

# 🚢 Project 1: Titanic Survival Prediction  
### Logistic Regression + Full Preprocessing Pipeline

This project demonstrates how pipelines manage the entire Titanic preprocessing workflow for binary classification.

## Pipeline Components

### 1. Feature Engineering
- Add `FamilySize`
- Add `IsAlone`
- Remove noisy or irrelevant features (PassengerId, Name, Cabin, Ticket, etc.)

### 2. Numeric Preprocessing
- Fill missing numeric values (median)
- Apply standard scaling

### 3. Categorical Preprocessing
- Fill missing categorical values (most frequent)
- One-Hot Encode all categories

### 4. Model
- Logistic Regression classifier
- Hyperparameter tuning using `RandomizedSearchCV`

## Key Insights
- Pipelines **eliminate data leakage** during cross-validation.
- Accuracy with pipeline is slightly lower but **more reliable** than manual workflows.
- Training time increases because preprocessing is repeated in every CV fold.
- Code becomes significantly cleaner and reusable.


# 🏡 Project 2: Ames Housing Price Prediction  
### Decision Tree Regressor + High-Dimensional Pipeline

The Ames dataset contains **79 features** with many missing values and categorical variables, making it an ideal dataset for pipelines.

## Why Ames Is a Perfect Pipeline Demonstration
- Many missing values → consistent imputation required  
- Many categorical variables → manual encoding is tedious  
- Generates a high-dimensional sparse matrix → pipelines handle it automatically  
- Highlights the power of **ColumnTransformer**

### 1. Feature Engineering
- Engineered new features (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`)
- Remove noisy or irrelevant features (`total_rooms`,`total_bedrooms`,`population`,`households`)

### 2. Numeric Preprocessing
- Fill missing numeric values (median)
- Apply standard scaling

### 3. Categorical Preprocessing
- Fill missing categorical values (most frequent)
- One-Hot Encode all categories

### 4. Model
- Decision Tree Regressor
- Hyperparameter tuning using `RandomizedSearchCV`

## Key Insights
- Pipelines avoid catastrophic mistakes like fitting imputation or encoding on the full dataset.
- They make it possible to handle 79 features with dozens of transformations safely.
- Decision Tree Regressor performed decently with R score = 0.736 
- Feature importances become reliable after preprocessing.
- RandomizedSearchCV may be slower, but provides a significantly better model.
- The final pipeline can be saved and deployed as one single object
---

## Requirements

```bash
pip install -r requirements.txt
```
---
