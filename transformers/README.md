# BERT Sentiment Classification

This project fine-tunes Google's BERT (bert-base-uncased) model for binary sentiment classification on the IMDB movie reviews dataset using Hugging Face Transformers.

## Overview

- **Model**: google-bert/bert-base-uncased
- **Task**: Sentiment Analysis (Positive/Negative)
- **Dataset**: IMDB (25,000 train, 25,000 test samples)
- **Framework**: PyTorch with Hugging Face Transformers

## Requirements

- Python 3.8+
- transformers
- datasets
- torch
- accelerate
- scikit-learn

Install dependencies:
```bash
pip install transformers[torch] datasets scikit-learn
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
)

# Fine-Tuning
trainer.train()
```

## Training Results

The model was trained for 3 epochs with early stopping. Below are the metrics per epoch:

| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall | F1     |
|-------|---------------|-----------------|----------|-----------|--------|--------|
| 1     | 0.216247     | 0.187472       | 0.925800 | 0.906977  | 0.949322 | 0.927666 |
| 2     | 0.113750     | 0.204079       | 0.932600 | 0.930186  | 0.935754 | 0.932962 |
| 3     | 0.048390     | 0.272328       | 0.936200 | 0.944693  | 0.926975 | 0.935750 |

- **Best Model**: Saved based on highest F1 score (Epoch 3).
- **Final Accuracy**: ~93.6% on validation set.

## Test Results

The fine-tuned model was evaluated on the full IMDB test set (50,000 samples) to assess generalization performance:

- **Accuracy**: 96.14%
- **Precision**: 96.25%
- **Recall**: 96.02%
- **F1 Score**: 96.13%
- **Loss**: 0.1644
- **Evaluation Runtime**: 539 seconds (92.76 samples/second)

### Classification Report

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| 0 (Negative) | 0.96     | 0.96  | 0.96    | 25,000 |
| 1 (Positive) | 0.96     | 0.96  | 0.96    | 25,000 |
| Macro Avg   | 0.96     | 0.96  | 0.96    | 50,000 |
| Weighted Avg| 0.96     | 0.96  | 0.96    | 50,000 |

The model demonstrates strong performance on unseen data, with balanced metrics across both classes.

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

- Training uses a 80/20 train/val split from the IMDB train set.
- Early stopping prevents overfitting with patience=2.
- For production, consider hyperparameter tuning or larger datasets.

For more details, refer to the `transformers.ipynb` notebook.