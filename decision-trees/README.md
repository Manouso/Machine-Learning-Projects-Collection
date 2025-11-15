# Decision Trees - Titanic Dataset

This project demonstrates how to use preprocessed data from the project data-preprocessing to predict target values by implementing decision trees
and compares this model with logistic regression to find which one performs the best. 

---

## Project Structure

- **decision-trees/**
  - `decision_trees.ipynb` — Main notebook for experiments  
  - `requirements.txt` — Requirements  
  - `README.md` — Project overview

---

## Objectives
 
- Use decision trees to classify passenger survival outcomes.
- Understand the concept of splitting criteria (Gini)
- Compare accuracy and f1-score with logistic regression
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

- The Decision Tree Classifier successfully classified passenger survival rate with strong accuracy 79%
- The features Sex and Pclass (passenger class) were by far the most most significant predictors.
- The Decision Tree Classifier nullified the significance of some features (Fare,Embarked,IsAlone) in contrast with Logistic Regression.
- The model’s F1-score confirmed a good trade-off between precision and recall, suggesting balanced predictive performance.
- The accuracy of Decision Tree Classifier (79%) is slightly worse than the accuracy of Logistic Regression (80%).
- Decision Tree Classifier finishes slightly faster than logistic regression (Decision Tree Classifier: 0.23s vs Logistic Regression: 0.26s)