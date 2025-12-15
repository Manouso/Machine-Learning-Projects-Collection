# MNIST MLP Classification Project

This project implements a complete machine learning pipeline for MNIST digit classification using PyTorch Multi-Layer Perceptron (MLP) with hyperparameter tuning and comprehensive evaluation.

## Overview

The notebook demonstrates:
- Data loading and exploratory data analysis (EDA) of MNIST dataset
- MLP model definition with configurable hidden layers, dropout, and batch normalization
- Hyperparameter tuning using RandomizedSearchCV with Stratified K-Fold cross-validation
- Training with validation monitoring, loss/accuracy curves, and learning rate scheduling
- Detailed evaluation metrics including accuracy, precision, recall, F1-score, and confusion matrix
- Sample predictions visualization

## Dataset

- **Training samples**: 60,000 (28x28 grayscale images)
- **Test samples**: 10,000 (28x28 grayscale images)
- **Classes**: 10 (digits 0-9)
- **Preprocessing**: Normalization to [-1, 1] range

## Model Architecture

MLP with:
- Input layer: 784 neurons (28x28 flattened)
- Hidden layers: Configurable (e.g., [128] or [128, 64])
- Output layer: 10 neurons (logits for 10 classes)
- Activation: ReLU for hidden layers
- Regularization: Dropout (0.2 or 0.3)
- Loss: CrossEntropyLoss

## Hyperparameter Tuning Results

Using RandomizedSearchCV with 5 iterations and 2-fold stratified cross-validation on 5,000 training samples:

- **Best Parameters**:
  - Optimizer: Adam
  - Learning Rate: 0.01
  - Hidden Sizes: [128]
  - Epochs: 5
  - Batch Size: 64
  - Dropout Prob: 0.2

## Performance Metrics

- **Previous Cross-Validation Score**: 0.8504 (85.04%)
- **Best Cross-Validation Score**: 0.902 (90.2%)

### Final Performance

- Test accuracy: 91.77% [previous version]
- Test accuracy: 97.22 [updated version]
- Training includes validation monitoring with loss/accuracy curves and StepLR scheduler

## Model Enhancements

Recent updates to improve model performance and stability:
- **Dropout**: Added dropout layers (0.2 or 0.3 probability) after ReLU activations to reduce overfitting
- **Batch Normalization**: Added batch normalization after each linear layer for stable training and faster convergence
- **Learning Rate Scheduler**: Integrated StepLR scheduler (gamma=0.9) in final training for adaptive learning rate decay


The enhancements provide better regularization and training stability, leading to improved performance.

## Key Features

- **Optimized for CPU**: Reduced dataset sizes and parameters for practical execution on consumer hardware
- **Advanced Regularization**: Dropout and batch normalization for better generalization
- **Adaptive Learning**: StepLR scheduler for optimal learning rate scheduling
- **Comprehensive Evaluation**: Classification report, confusion matrix, and sample predictions
- **Visualization**: Training curves, class distributions, sample images
- **Modular Design**: Sklearn-compatible wrapper for easy hyperparameter tuning

## Dependencies

- torch
- torchvision
- scikit-learn
- matplotlib
- seaborn
- numpy

## Results Summary

The enhanced MLP achieves strong performance on MNIST with improved training stability and generalization. The hyperparameter tuning identifies optimal settings for architecture and training parameters, resulting in reliable digit classification with high accuracy. The addition of dropout, batch normalization, and learning rate scheduling further improves the model's robustness and performance.
