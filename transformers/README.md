# BERT Sentiment Classification

This project fine-tunes Google's BERT (bert-base-uncased) model for binary sentiment classification on the IMDB movie reviews dataset using Hugging Face Transformers. It includes a baseline model and an optimized version with improved performance and speed.

## Overview

- **Model**: google-bert/bert-base-uncased
- **Task**: Sentiment Analysis (Positive/Negative)
- **Dataset**: IMDB (25,000 train, 25,000 test samples for training/validation; downloaded CSV for final testing)
- **Framework**: PyTorch with Hugging Face Transformers
- **Optimizations**: Custom optimizer, dropout, early stopping, FP16, gradient accumulation

## Requirements

- Python 3.8+
- transformers
- datasets
- torch
- accelerate
- scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup and Training

1. Open `transformers.ipynb` in Jupyter Notebook or VS Code.
2. Run the cells sequentially to load data, tokenize, and train the model.
3. The notebook includes:
   - Data loading and preprocessing
   - Model fine-tuning with early stopping
   - Evaluation on validation and test sets

Key training code:
```python
# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    optimizers=(optimizer, None),  # Custom optimizer
)

# Fine-Tuning
trainer.train()
```
## Model Comparison

### Baseline Model (Simple BERT)
Trained for 3 epochs with default settings.

**Training Results** (per epoch):

| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall | F1     |
|-------|---------------|-----------------|----------|-----------|--------|--------|
| 1     | 0.216247     | 0.187472       | 0.925800 | 0.906977  | 0.949322 | 0.927666 |
| 2     | 0.113750     | 0.204079       | 0.932600 | 0.930186  | 0.935754 | 0.932962 |
| 3     | 0.048390     | 0.272328       | 0.936200 | 0.944693  | 0.926975 | 0.935750 |

- **Best Model**: Epoch 3 (F1: 0.935750)
- **Validation Accuracy**: 93.62%

**Test Results** (on IMDB test set):
- **Accuracy**: 96.14%
- **Precision**: 96.25%
- **Recall**: 96.02%
- **F1 Score**: 96.13%
- **Loss**: 0.1644

### Optimized Model (Improved BERT)
Trained for up to 10 epochs with optimizations: batch size 32, gradient accumulation 2, dropout 0.2, warmup 100, early stopping.

**Training Results** (per epoch):

| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall | F1     |
|-------|---------------|-----------------|----------|-----------|--------|--------|
| 1     | 0.890058     | 0.420908       | 0.918960 | 0.940677  | 0.894320 | 0.916913 |
| 2     | 0.832549     | 0.413913       | 0.923480 | 0.891386  | 0.964480 | 0.926494 |
| 3     | 0.776441     | 0.409009       | 0.933160 | 0.925836  | 0.941760 | 0.933730 |
| 4     | 0.758693     | 0.408799       | 0.935680 | 0.935471  | 0.935920 | 0.935695 |
| 5     | 0.720316     | 0.415885       | 0.933360 | 0.947021  | 0.918080 | 0.932326 |
| 6     | 0.707563     | 0.418805       | 0.935400 | 0.929660  | 0.942080 | 0.935829 |


- **Best Model**: Epoch 6 (F1: 0.935829, early stopping triggered)
- **Validation Accuracy**: 93.54%

**Test Results** (on downloaded CSV test set):
- **Accuracy**: 96.32%
- **Precision**: 95.96%
- **Recall**: 96.70%
- **F1 Score**: 96.33%
- **Loss**: 0.3794

**Comparison Summary**:
- Baseline: 96.14% test accuracy, faster training (3 epochs).
- Optimized: 96.32% test accuracy (+0.18%), better generalization, but longer training (8 epochs). Optimizations improve performance slightly with less overfitting.

## Usage

After training, use the model for inference:
```python
from transformers import pipeline

# Load the fine-tuned model
classifier = pipeline("sentiment-analysis", model="./bert-sentiment")

# Predict
result = classifier("This movie was amazing!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.99}]
```

## Notes

- Early stopping prevents overfitting with patience=2.
- For production, consider hyperparameter tuning or larger datasets.

For more details, refer to the `transformers.ipynb` notebook.