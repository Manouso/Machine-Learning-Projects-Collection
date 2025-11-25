# LightGBM Heart Disease Classification

Complete machine learning pipeline using LightGBM for binary classification of heart disease with comprehensive model evaluation and hyperparameter optimization.

## Objectives

- **End-to-end ML workflow** from data preprocessing to model deployment
- **Advanced hyperparameter tuning** using RandomizedSearchCV (20 iterations)
- **Comprehensive evaluation** with 6 metrics and visual analysis
- **Model interpretability** through feature importance analysis
- **Best practices** in stratified sampling, cross-validation, and model persistence

## Dataset

**Heart Disease UCI** - 918 patients, 11 clinical features
- Binary classification: Heart disease presence (Yes/No)
- Features: Age, Sex, Chest Pain, BP, Cholesterol, ECG, Heart Rate, etc.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook lightgbm.ipynb

# Run all cells sequentially
```

## Project Structure

- **lightgbm/**
  - `lightgbm.ipynb` - Notebook
  - `requirements.txt` — Requirements  
  - `README.md` — Project overview

---

## Pipeline Overview

The notebook implements an 11-step ML workflow:

1. **Data Loading & EDA** - Statistical analysis and 6 visualizations
2. **Preprocessing** - Encoding, imputation, feature scaling
3. **Train-Test Split** - 80/20 stratified sampling
4. **Baseline Model** - Default LightGBM parameters
5. **Hyperparameter Tuning** - RandomizedSearchCV (50 combinations)
6. **Model Evaluation** - Confusion matrix, ROC curves, metrics comparison
7. **Feature Importance** - Identify top predictive features
8. **Predictions** - Confidence analysis and sample cases
9. **Model Persistence** - Save artifacts for deployment

---

## Performance Results
      
    Metric  Baseline  Optimized  Improvement  Improvement %
 Accuracy  0.842391   0.891304     0.048913           5.81
Precision  0.884211   0.901961     0.017750           2.01
   Recall  0.823529   0.901961     0.078431           9.52
 F1-Score  0.852792   0.901961     0.049169           5.77
  ROC-AUC  0.906743   0.929579     0.022836           2.52
      MCC  0.685663   0.780010     0.094346          13.76

**Top Predictive Features:**
1. ST_Slope (slope pattern)
2. ChestPainType (pain characteristics)
3. Oldpeak (ST depression)
4. MaxHR (maximum heart rate)
5. ExerciseAngina (exercise-induced)

---

## Technical Implementation

**Model:** LightGBM Classifier
- Gradient-based One-Side Sampling (GOSS)
- Leaf-wise tree growth
- Histogram-based learning

**Validation:** 5-Fold Stratified Cross-Validation

---

## Key Highlights

 **Complete Pipeline** - From raw data to deployment-ready model  
 **Optimized Performance** - 93% ROC-AUC through systematic tuning    
 **Interpretable** - Feature importance and confidence analysis  
 **Best Practices** - Stratified sampling, CV, baseline comparison  


## References

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- Ke et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"

---

**Educational ML project demonstrating industry-standard practices in gradient boosting classification**
