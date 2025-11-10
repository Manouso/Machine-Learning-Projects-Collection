# Logistic Regression - Titanic Dataset

This project demonstrates how to use preprocessed data from the project data-preprocessing to predict target values by implementing logistic regression on the data.

---

## Project Structure

logistic-regression/
│
├── logistic_regression.ipynb   # Main notebook
├── requirements.txt
└── README.md


---

## Objectives

- Use logistic regression to classify passenger survival outcomes.
- Evaluate performance using Accuracy, Precision, Recall, and F1-score.
- Interpret feature importance to understand which factors most influenced survival.

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

## Requirements

```bash
pip install -r requirements.txt
```
---

## Insights

