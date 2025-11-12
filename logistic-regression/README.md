# Logistic Regression - Titanic Dataset

This project demonstrates how to use preprocessed data from the project data-preprocessing to predict target values by implementing logistic regression on the data and
applying hyperparameter-tuning and cross-validation to improve model performance.

---

## Project Structure

logistic-regression/
│
├── logistic_regression.ipynb # Main notebook
├── requirements.txt
└── README.md

---

## Objectives

- Use logistic regression to classify passenger survival outcomes.
- Use cross-validation and hyperparameter tuning to improve model performance.
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

- The Logistic Regression model successfully classified passenger survival with strong accuracy 80%.
- Gender (Sex) and Passenger Class (Pclass) were the most significant predictors (and coeffiecients of the model).
- Fare showed positive relationship with survival.
- FamilySize had a moderate and negative influence on survival but being alone had a mild and negative influence as well.
- Age and Embarked had weaker effects, though younger passengers had higher survival rate.
- The model’s F1-score confirmed a good trade-off between precision and recall, suggesting balanced predictive performance.
- Random Search has better accuracy than GridSearch in our example (RandomSearch:80% vs GridSearch:79%).
- Random Search finishes way faster than GridSearch (RandomSearch: 0.26s vs GridSearch: 2.82s).