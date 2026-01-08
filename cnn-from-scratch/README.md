# CNN from Scratch with PyTorch

This project implements a Convolutional Neural Network (CNN) using PyTorch for classifying the CIFAR-10 dataset. The CNN uses PyTorch's nn.Module for layers, with data augmentation and hyperparameter tuning.

## Features

- CNN model using PyTorch nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.Linear, nn.Dropout
- Data preprocessing: normalization and data augmentation (random crop, horizontal flip, random erasing, Gaussian noise)
- Hyperparameter search over learning rate, batch size, dropout rate, weight decay
- Training with Adam optimizer
- Evaluation on test set with confusion matrix and classification report
- GPU support if available

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Jupyter notebook `cnn_from_scratch.ipynb`.
2. The notebook will download CIFAR-10 automatically if not present.
3. It performs hyperparameter search on a subset of epochs to find the best combo.
4. Then trains the model with the best hyperparameters for 10 epochs.
5. Evaluates on the test set and plots results.

## Data Preprocessing

- **Normalization**: Images normalized using CIFAR-10 mean and std per channel.
- **Data Augmentation** (training only): Random crop (with padding), random horizontal flip, random erasing, Gaussian noise.

## Model Architecture

The CNN consists of 4 convolutional blocks followed by fully connected layers:

**Convolutional Blocks**:
- Conv2D (3 → 32, 3x3, padding=1) + BatchNorm2D + ReLU + MaxPool2D (2x2) → 16x16x32
- Conv2D (32 → 64, 3x3, padding=1) + BatchNorm2D + ReLU + MaxPool2D (2x2) → 8x8x64
- Conv2D (64 → 128, 3x3, padding=1) + BatchNorm2D + ReLU + MaxPool2D (2x2) → 4x4x128
- Conv2D (128 → 256, 3x3, padding=1) + BatchNorm2D + ReLU + MaxPool2D (2x2) → 2x2x256

**Fully Connected Layers**:
- Flatten → 1024 features
- Linear (1024 → 512) + ReLU + Dropout
- Linear (512 → 10 classes)

Default filter sizes: 32, 64, 128, 256 (configurable in CustomCNN class).

## Hyperparameters

Tuned via 5-fold cross-validation grid search:

- Learning rate: [0.0001, 0.0005, 0.001, 0.005]
- Batch size: [32, 64]
- Dropout rate: [0.2, 0.3, 0.4]
- Weight decay: [0, 1e-5, 1e-4]
- Optimizer: Adam
- Total combinations: 72 (4 × 2 × 3 × 3)

Training uses LR scheduler (StepLR), early stopping, mixed precision for GPU optimization, and label smoothing.

## Actual Performance

**Test Accuracy**: 85.77%

**Best Model Hyperparameters**: 
- Learning Rate: 0.001
- Batch Size: 32
- Dropout Rate: 0.2
- Weight Decay: 0

**Total Runtime**: ~17 hours

## Notes

- Uses PyTorch for efficient computation and autograd.
- Supports GPU if available.
- Hyperparameter search trains each combo for 2 epochs to save time.
