# Extreme Gradient Boosting

This project demonstrates how to train and evaluate a **ExtremeGradientBoostingClassifier** using scikit-learn.

---

## Objectives

- Train and optimize an XGBoostClassifier using scikit-learn and the XGBoost library.
- Understand why XGBoost often outperforms classical Gradient Boosting.
- Use Pipeline and ColumnTransformer for clean preprocessing.
- Extracting and visualizing **feature importances**.
- Evaluating  performance with accuracy / confusion matrix.

---

## Project Structure

- **extreme-gradient-boosting/**
  - `xgradient_boosting.ipynb` - Notebook
  - `heart.csv` - Dataset
  - `requirements.txt` — Requirements  
  - `README.md` — Project overview

---

## Dataset

[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

This dataset contains 918 patient records, each with 11 clinical features that are commonly used in cardiovascular diagnostics.
The goal is to predict whether a patient is likely to develop heart disease based on medical measurements, lifestyle indicators, and symptoms.

---

## Requirements

```bash
pip install -r requirements.txt
```
---

## Insights

- The Extreme Gradient Boosting Classifier demonstrates robust predictive performance, achieving approximately 90% accuracy and an ROC-AUC score of 0.92 prior to feature engineering.
- After feature engineering, performance remained unchanged, indicating that most features already carry substantial predictive power and that XGBoost’s internal structure handles redundant features effectively.
- The only observed change was a slight reduction in runtime; however, this difference is negligible for real-world applications, where inference speed and model interpretability are more critical than training speed on small datasets.
- Comparatively, the Extreme Gradient Boosting Classifier was slightly less accurate than the Random Forest Classifier, for which feature engineering did provide some performance gains, suggesting that Gradient Boosting is more robust to redundant or low-importance features.
- Comparatively the Extreme Gradient Boosting Classifier was slightly more accurate that the post feature engineering version of Gradient Boosting, for which feature engineering did provide some performance gains, suggesting that Gradient Boosting is more robust to redundant or low-importance features.
- Overall, the Extreme Gradient Boosting Classifier provides a stable, high-performing baseline, and additional performance gains would likely require either more advanced ensemble techniques or additional predictive features.