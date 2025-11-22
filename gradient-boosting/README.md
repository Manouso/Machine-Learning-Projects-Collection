# Gradient Boosting

This project demonstrates how to train and evaluate a **GradientBoostingClassifier** using scikit-learn.

---

## Objectives

- Training a GradientBoostingClassifier.
- Learn why Boosting performs so well and compare it with RandomForestClassifier.
- Using Pipelines for preprocessing and modeling.
- Extracting and visualizing **feature importances**.
- Evaluating  performance with accuracy / confusion matrix.

---

## Project Structure

- **gradient-boosting/**
  - `gradient_boosting.ipynb` - Notebook
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

- The Gradient Boosting Classifier demonstrates robust predictive performance, achieving approximately 90% accuracy and an ROC-AUC score of 0.92 prior to feature engineering.
- Post feature engineering, there was no measurable improvement in either accuracy or ROC-AUC, indicating that the original set of features already captures the majority of predictive signal in the dataset.
- The only observed change was a slight reduction in runtime; however, this difference is negligible for real-world applications, where inference speed and model interpretability are more critical than training speed on small datasets.
- Comparatively, the Gradient Boosting Classifier was slightly less accurate than the Random Forest Classifier, for which feature engineering did provide some performance gains, suggesting that Gradient Boosting is more robust to redundant or low-importance features.
- Overall, the Gradient Boosting Classifier provides a stable, high-performing baseline, and additional performance gains would likely require either more advanced ensemble techniques or additional predictive features.