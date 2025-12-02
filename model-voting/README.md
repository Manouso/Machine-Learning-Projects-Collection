# Model Voting Credit Card Fraud Detection

Ensemble voting classifier combining XGBoost, Random Forest, and Logistic Regression optimized for high recall in detecting fraudulent credit card transactions.

## Objectives

- **Fraud detection pipeline** optimized for recall and F1-score
- **Two-stage hyperparameter tuning** with RandomizedSearch and Bayesian Optimization
- **Class imbalance handling** with aggressive penalty weights for missed frauds
- **Comprehensive evaluation** focused on minimizing false negatives

## Dataset

**Credit Card Fraud Detection** - Kaggle dataset
- 284,807 transactions with extreme class imbalance (0.17% fraud)
- 30 features: Time, Amount, and 28 PCA-transformed features (V1-V28)
- Binary classification: Fraud (1) vs. Legitimate (0)
- Sampled to 30% for faster training (85,442 transactions)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook model_voting.ipynb

# Run all cells sequentially
```

## Project Structure

- **model-voting/**
  - `model_voting.ipynb` - Complete implementation
  - `README.md` - Documentation

---

## Pipeline Overview

**Voting Architecture:**
- **Base Learners:** Logistic Regression, Random Forest, XGBoost
- **Voting Strategy:** Soft voting (probability-based)
- **Preprocessing:** StandardScaler + SimpleImputer
- **Validation:** 2-fold Stratified Cross-Validation
- **Optimization:** RandomizedSearch → Bayesian Optimization

---

## Performance Results

| Metric     | RandomSearch | Bayesian | Improvement |
|------------|--------------|----------|-------------|
| **Recall** | 76.67%       | 83.33%   | **+6.67%**  |
| **F1-Score**| 82.14%      | 84.75%   | **+2.60%**  |
| Precision  | 88.46%       | 86.21%   | -2.25%      |

---

## Technical Approach

### Two-Stage Optimization

**Stage 1: RandomizedSearch (Broad Exploration)**
- Scoring: Recall (prioritize catching frauds)
- 10 iterations exploring wide parameter ranges
- Focus: Find promising hyperparameter regions

**Stage 2: Bayesian Optimization (Fine-Tuning)**
- Scoring: F1 (balance recall and precision)
- 15 iterations in narrow ranges around RandomSearch best params
- Focus: Optimize recall while maintaining acceptable precision

### Class Imbalance Handling

**Penalty Weights:**
- Logistic Regression: `class_weight='balanced'`
- Random Forest: `class_weight='balanced'`
- XGBoost: `scale_pos_weight=20-150` (fraud is 20-150x more important)

---

## Key Hyperparameters

**Most Impactful for Recall:**
1. **scale_pos_weight** (XGBoost): 50-150 range
2. **class_weight** (Logistic/RF): 'balanced'
3. **learning_rate** (XGBoost): 0.01-0.3
4. **C** (Logistic): 0.05-10

**Focused Ranges (Bayesian):**
- Built ±50% ranges around RandomSearch best values
- Example: If best `scale_pos_weight=50`, search 25-75

---

## Model Comparison

### RandomizedSearch
- **Strengths:** Fast exploration, good starting point
- **Recall:** 76.67% (7 missed frauds)
- **F1:** 82.14%

### Bayesian Optimization  
- **Strengths:** Intelligent fine-tuning, better recall
- **Recall:** 83.33% (5 missed frauds)  
- **F1:** 84.75%
- **Winner:** Best overall performance

---

## Key Highlights

 **High Recall** - 83.33% fraud detection rate  
 **Two-Stage Optimization** - RandomizedSearch → Bayesian  
 **Extreme Imbalance Handled** - 0.17% fraud successfully detected  
 **Business-Focused** - Heavy penalty for missed frauds

## Technical Challenge
**Challenge : Extreme Class Imbalance (577:1 ratio)**
- Solution: Aggressive class weights and scale_pos_weight parameters
- Result: 83.33% recall despite only 0.17% fraud prevalence

---

## References

- [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [XGBoost scale_pos_weight](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [Scikit-Optimize BayesSearchCV](https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html)

---

**Educational ML project demonstrating ensemble voting with advanced hyperparameter optimization for imbalanced fraud detection**
