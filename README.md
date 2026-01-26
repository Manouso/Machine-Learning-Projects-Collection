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
- Implemented Learning and Validation Curves to gain insight about model's performance.
- Visualized Error Analysis of features to see how much do they affect the model negatively. 
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


## 11. Light Gradient Boosting Classifier with Pipelines on Heart Disease Data (with new advanced practices)
- Built a Light Gradient Boosting Classifier pipeline for the **Heart Failure Prediction Dataset**. 
- Built a new more complete and advanced EDA for the **Heart Failure Prediction Dataset**.
- Performed Hyperparameter Tuning and used Stratified Cross Validation.
- Evaluate feature importance prior to feature engineering to receive necessary insight.
- Performed Feature Engineering by engineering new features with knowledge from the healthcare domain.
- Dataset: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

## 12. Model Stacking with Pipelines on Heart Disease Data
- Built a Model Stacking pipeline for basic and meta models.
- Built advanced EDA to capture the characteristics of  **Heart Failure Prediction Dataset**.
- Performed Hyperparameter Tuning and used Stratified Cross Validation.
- Evaluate feature importance prior to feature engineering to receive necessary insight.
- Performed Feature Engineering by engineering new features in limited number however due to overfitting issues.
- Dataset: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

## 13. Model Voting with Pipelines on Fraud Detection Data
- Built a Model Voting pipeline which combines the strengths of 3 ML models.
- Built advanced EDA to capture the characteristics of **Credit Card Fraud Detection Dataset**.
- Performed more advanced Hyperparameter Tuning by combining Random Search with Bayesian Optimization.
- Evaluate feature importance prior to feature engineering to receive necessary insight.
- Make an Error Analysis to gain insight about the mistakes and the blind spots of the model.
- Performed Feature Engineering by engineering new features to match the complexity of the new problem.
- Dataset: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## 14. FastAPI Credit Card Fraud Prediction Service
- Built a production-ready REST API using FastAPI to serve real-time predictions from an ensemble voting classifier.
- Implemented proper input validation using Pydantic models for the 30 credit card transaction features (Time, Amount, V1-V28).
- Integrated model loading and prediction logic with error handling for missing model files and prediction failures.
- Created comprehensive API documentation with interactive Swagger UI for easy testing and integration.
- Handled pickle deserialization issues by properly importing feature engineering functions used during model training.
- Implemented response formatting with both binary predictions and probability scores for fraud detection confidence.
- Dataset: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### 15. Neural Network from Scratch with NumPy
- Implemented a complete 2-layer neural network from scratch using NumPy for binary classification on the `make_circles` dataset.
- Included forward propagation, backpropagation, and gradient descent optimization with ReLU and Sigmoid activations.
- Demonstrated proper weight initialization (He), loss tracking, and evaluation metrics (accuracy, confidence).
- Dataset: Scikit-learn `make_circles` (synthetic 2D dataset with 100 samples).

### 16. Neural Network with PyTorch on Titanic Dataset
- Built a 2-layer neural network using PyTorch `nn.Module` for binary classification on the Titanic survival dataset.
- Integrated advanced sklearn pipelines with `FunctionTransformer` for missing values handling and feature engineering.
- Implemented reproducible training with standardized random states across NumPy, PyTorch, and sklearn.
- Used mini-batch training with PyTorch DataLoader and evaluated with accuracy/classification metrics.
- Dataset: Kaggle Titanic dataset with engineered features (FamilySize, Title, AgeBin, FareBin).

### 17. Multi-Layer Perceptron on MNIST Dataset
- Implemented an MLP using PyTorch for MNIST digit classification with hyperparameter tuning.
- Included configurable hidden layers, dropout, and batch normalization.
- Used RandomizedSearchCV with Stratified K-Fold cross-validation for tuning.
- Trained with validation monitoring, loss/accuracy curves, and learning rate scheduling.
- Evaluated with accuracy, precision, recall, F1-score, confusion matrix, and sample predictions.
- Dataset: MNIST (70,000 28x28 grayscale images of digits 0-9).

### 18. Convolutional Neural Network on MNIST Dataset
- Built a modern CNN using PyTorch for handwritten digit recognition on MNIST.
- Incorporated batch normalization, dropout, and data augmentation (random rotation and translation).
- Utilized mixed precision training (FP16) for faster training on GPUs.
- Performed hyperparameter tuning with random search and early stopping to prevent overfitting.
- Evaluated with comprehensive metrics including classification reports and confusion matrices.
- Dataset: MNIST (70,000 28x28 grayscale images of digits 0-9).

### 19. CNN from Scratch on CIFAR-10 Dataset
- Implemented a Convolutional Neural Network from scratch using PyTorch for CIFAR-10 image classification.
- Included data preprocessing with normalization and augmentation (random crop, horizontal flip, random erasing, Gaussian noise).
- Performed hyperparameter tuning for learning rate, batch size, dropout, and weight decay.
- Used Adam optimizer with GPU support and evaluated with confusion matrix and classification report.
- Dataset: CIFAR-10 (60,000 32x32 color images in 10 classes).

### 20. Transfer Learning with ResNet18 on CIFAR-10
- Demonstrated transfer learning using pretrained ResNet18 on CIFAR-10 with progressive unfreezing.
- Applied extensive data augmentation (random cropping, flipping, rotation, color jittering, affine transformations, perspective distortion, random erasing).
- Replaced the classifier head with a custom multi-layer network including dropout and batch normalization.
- Implemented progressive unfreezing: trained classifier first, then unfroze later layers for fine-tuning.
- Performed hyperparameter optimization with random search over learning rates, batch sizes, and unfreezing strategies.
- Achieved best validation accuracy of 58.56% with detailed evaluation metrics.
- Dataset: CIFAR-10 (60,000 32x32 color images in 10 classes).

### 21. EfficientNet-B1 Transfer Learning on CIFAR-10
- Implemented EfficientNet-B1 with a hyperparameter-tuned classifier head for CIFAR-10 classification.
- Froze the pretrained backbone initially, training only the classifier, then optionally fine-tuned the backbone.
- Used advanced data augmentation techniques (random crop, flip, rotation, color jitter, affine transformations, random erasing).
- Performed random search over learning rates, weight decay, dropout, optimizers, and schedulers.
- Evaluated on train/validation/test splits with detailed metrics and visualizations.
- Dataset: CIFAR-10 (60,000 32x32 color images in 10 classes).

---

## Tools & Libraries
- Python
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn
- PyTorch
- KaggleHub
- XGBoost
- LightGBM

---

## Goals
- Build a structured foundation in data analysis and machine learning.
- Apply theoretical knowledge through small, focused projects.
- Prepare datasets for modeling using professional preprocessing techniques.
- Create a portfolio that demonstrates practical ML and data handling skills.

---


