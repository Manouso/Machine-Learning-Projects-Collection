# Pipeline Implementation Project

This project demonstrates how to build, evaluate, and compare **scikit-learn Pipelines** for a Logistic Regression model using the Titanic dataset.  
It highlights why pipelines are essential in real machine learning workflows, focusing on clean structure, preventing data leakage, and simplifying experimentation.

---

## Project Structure

- **pipeline/**
  - `notebooks`
    - `pipeline_logistic_regression.ipynb` — Main notebook for experiments
  - `data`
      `Titanic-Dataset.csv`  — The dataset
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

## Dataset

- Columns: 
   - `Survived`: Index of survival (1 = survived, 0 = didn't survived)     
   - `Pclass`: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)            
   - `Sex`: Male/Female         
   - `Age`: Age in years         
   - `FamilySize`: How many family members were aboard            
   - `Fare`: Passenger Fare       
   - `Cabin`: Number of Cabin       
   - `Embarked`: Port of embarkation (C, Q, S) 
   - `IsAlone`: Binary indicator for traveling alone (1 = alone, 0 = not alone)

---

## Pipeline Components

The pipeline performs the following steps:

### **1. Feature Engineering**
- Add `FamilySize`  
- Add `IsAlone`
- Remove `PassengerId`, `Name`, `Ticket`, `Cabin`,`SibSp`, `Parch`

### **2. Column Preprocessing**
- **Numeric pipeline**  
  - Fill missing values using median  
  - StandardScaler  
- **Categorical pipeline**  
  - Fill missing values using most frequent  
  - OneHotEncoder  

### **3. Model**
- Logistic Regression  
- Hyperparameter tuning (`C`, solver, etc.)

---

## Requirements

```bash
pip install -r requirements.txt
```
---

## Insights

- Pipelines ensure all preprocessing and feature engineering steps are applied consistently, avoiding data leakage during cross-validation.
- Accuracy was a slightly lower compared to manual implementation (79.9% vs 80.04%), but that accuracy may be more represantative.
- Training time of the new implementation was higher (0.95 sec vs 0.26sec) and that can be explained as 
  Pipelines may take longer to fit due to preprocessing being applied in each fold of cross-validation.
- Overall, pipelines make the coding easier and most important of all it prevents data leakage and errors in preprocessing.