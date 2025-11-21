# Random Forest Classifier

This project demonstrates how to train and evaluate a **RandomForestClassifier** using scikit-learn.

---

## Objectives

- Training a RandomForestClassifier
- Using Pipelines for preprocessing and modeling
- Extracting and visualizing **feature importances**
- Evaluating  performance with accuracy / confusion matrix

---

## Project Structure

- **random-forest-classifier/**
  - `random_forest.ipynb` - Notebook
  - `Housing-Dataset.csv` - Dataset
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

- Random Forest Classifier delivers the strongest performance, achieving an accuracy of 90% and an ROC-AUC score of 0.925 before feature engineering.
- This iteration introduced Mutual Information to assess how strongly each feature relates to the target and to detect inter-feature dependency. 
- Incorporating this analysis into the feature engineering process proved highly valuable—after removing low-information 
  and redundant features, model performance improved to 91% accuracy and an ROC-AUC score of 0.931.
- Feature engineering not only enhanced predictive performance but also reduced overall training time, from 5.08 seconds down to 4.93 seconds.
- Unlike previous versions where feature importance was used only post-training for evaluation, this workflow used feature importance proactively, guiding the feature-selection phase for more informed model refinement.
