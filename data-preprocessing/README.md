# Data Preprocessing & Feature Engineering

This project demonstrates how to prepare a dataset for machine learning by handling missing values, encoding categorical features, scaling numeric variables, and engineering the features to improve model performance.

---

## Project Structure

- **data-preprocessing/**
  - `data_preprocessing.ipynb` — Main notebook for experiments  
  - `requirements.txt` — Requirements  
  - `README.md` — Project overview

---


## Objectives

- Identify and handle any missing values correctly.
- Encode categorical features.
- Scale numerical variables using standardization.
- Engineer new features to enhance model performance.
- Produce a ready-to-use dataset for ML models.

---

## Dataset

- Source: [Titanic Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- Columns:
   - `PassengerId`: Unique ID for each passenger 
   - `Survived`: Index of survival (1 = survived, 0 = didn't survived)     
   - `Pclass`: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)   
   - `Name`: Passenger Name         
   - `Sex`: Male/Female         
   - `Age`: Age in years         
   - `SibSp`: Number of siblings or spouses aboard     
   - `Parch`: Number of parents or children aboard       
   - `Ticket`: Ticket Number     
   - `Fare`: Passenger Fare       
   - `Cabin`: Number of Cabin       
   - `Embarked`: Port of embarkation (C, Q, S) 
---

## Requirements

```bash
pip install -r requirements.txt
```
---

## Insights
- Missing values handled by median imputation for `Age` and mode for `Embarked`.
- Created `FamilySize` = `SibSp` + `Parch` + 1 and dropped redundant features.
- Converted categorical features (`Sex`, `Embarked`) using one-hot encoding.
- Scaled features to standardize their range and variance for machine learning algorithms.
- Implemented EDA to see correlation between features and removed high correlation features (`AgeGroup`, `SibSp`, `Parch`).
- `Fare`, `Age`, `FamilySize` log-transformed to reduce right skew for machine learning algorithms. 