# Machine Learning Mini Projects

A collection of data science and machine learning mini projects built to strengthen core concepts — from linear regression to real-world data preprocessing and exploratory data analysis.

---

## Projects Included

### 1. Linear Regression Basics
- Implemented linear regression from scratch using NumPy and compared results with scikit-learn’s `LinearRegression`.
- Demonstrated understanding of model training, prediction, and evaluation (MSE, R²).
- Dataset: Salary vs Experience data from Kaggle.

### 2. Advertising Dataset EDA
- Performed exploratory data analysis on an advertising dataset.
- Identified correlations between marketing channels and sales.
- Created clear visualizations using Matplotlib and Seaborn.
- Dataset: Kaggle Advertising dataset.

### 3. Titanic Data Preprocessing & Feature Engineering
- Cleaned and preprocessed the Titanic dataset for machine learning.
- Handled missing values, encoded categorical variables, and scaled continuous features.
- Engineered new features (`FamilySize`, `IsAlone`) and analyzed their effect on survival.
- Dataset: Kaggle Titanic dataset.

### 4. Logistic Regression on Pre-Processed Titanic Data
- Implemented Logistic Regression on the pre-processed data from the previous project.
- Used cross-validation to improve model's training.
- Applied Hyperparameter tuning to find the best hyperparameters to improve model's performance.
- Visualized the importance of model's features and ordered based on their coefficients.
- Dataset: Kaggle Titanic dataset.

### 5. Decision Trees on Pre-Processed Titanic Data
- Built a Decision Tree classifier on the preprocessed Titanic dataset.
- Explored the effect of `max_depth`, `min_samples_split`, and `min_samples_leaf` on model performance.
- Visualized decision tree structure and feature importance.
- Dataset: Kaggle Titanic dataset.

### 6. Pipelines with Logistic Regression on Titanic Data
- Built a full scikit-learn pipeline including:
  - Feature engineering
  - Column dropping
  - Numeric preprocessing
  - Categorical preprocessing
  - Logistic Regression classifier
- Integrated hyperparameter tuning inside the pipeline to avoid data leakage.
- Measured model accuracy and training time with cross-validation.
- Dataset: Kaggle Titanic dataset.


### 7. Pipelines with Decision Trees on Housing Data
- Built a Decision Tree Regression pipeline for the **Ames Housing dataset**.
- Handled missing values, one-hot encoded categorical variables, and scaled numeric features inside the pipeline.
- Performed hyperparameter tuning (`max_depth`, `min_samples_split`, `min_samples_leaf`) using cross-validation.
- Evaluated model performance using RMSE and visualized feature importances.
- Dataset: [Ames Housing Dataset (CSV)](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset)

### 8. Random Forest Classifier with Pipelines on Heart Disease Data
- Built a Random Forest Tree Classifier pipeline for the **Heart Failure Prediction Dataset**. 
- One-hot encoded categorical variables and scaled numeric features using StandardScaler.
- Performed hyperparameter tuning and used Stratified Cross Validation.
- Performed Feature Engineering by dropping non-important features of the dataset.
- Dataset: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

### 9. Gradient Boosting Classifier with Pipelines on Heart Disease Data
- Built a Gradient Boosting Classifier pipeline for the **Heart Failure Prediction Dataset**. 
- Used Explanatory Data Analysis of the Heart Disease Data from the **Random Forest Classifier** project.
- Performed hyperparameter tuning and used Stratified Cross Validation.
- Evaluate feature importance prior to feature engineering to receive necessary insight.
- Performed Feature Engineering by dropping non-important features of the dataset using a threshold of importance.
- Dataset: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

### 10. Extreme Gradient Boosting Classifier with Pipelines on Heart Disease Data
- Built an Extreme Gradient Boosting Classifier pipeline for the **Heart Failure Prediction Dataset**. 
- Used Explanatory Data Analysis of the Heart Disease Data from the **Random Forest Classifier** project.
- Performed hyperparameter tuning and used Stratified Cross Validation.
- Evaluate feature importance prior to feature engineering to receive necessary insight.
- Performed Feature Engineering by dropping non-important features of the dataset using a threshold of importance.
- Dataset: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
---

## Tools & Libraries
- Python
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn
- KaggleHub

---

## Goals
- Build a structured foundation in data analysis and machine learning.
- Apply theoretical knowledge through small, focused projects.
- Prepare datasets for modeling using professional preprocessing techniques.
- Create a portfolio that demonstrates practical ML and data handling skills.

---


