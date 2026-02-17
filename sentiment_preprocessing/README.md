# IMDB Sentiment Text Preprocessing

This project demonstrates text preprocessing techniques for sentiment analysis on the IMDB movie reviews dataset. It includes two approaches: classical machine learning methods using Bag-of-Words (BoW) and TF-IDF vectorization, and neural network preparation with vocabulary building and sequence conversion. The project also includes model training and evaluation for both approaches, with comparisons.

## Features

### Classical Approach (`classical-sentiment-preprocessing.ipynb`)
- **Text Preprocessing**: Lemmatization, stopword removal, and punctuation filtering using spaCy.
- **Vectorization**: BoW and TF-IDF with n-gram support (unigrams and bigrams).
- **Model Training**: Logistic Regression and Naive Bayes classifiers with cross-validation.
- **Evaluation**: Classification reports and confusion matrices.

### Neural Approach (`neural-sentiment-preprocessing.ipynb`)
- **Text Preprocessing**: Lemmatization, stopword removal, and punctuation filtering using spaCy.
- **N-gram Generation**: Create bigrams from cleaned text.
- **Vocabulary Building**: Construct a vocabulary with special tokens (`<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`) and word-to-index mappings.
- **Sequence Conversion**: Transform text into numerical sequences for neural model input.
- **Embedding Preparation**: Load GloVe embeddings and create an embedding matrix.
- **Model Training**: PyTorch-based neural classifier with embedding layer, fully connected layers, dropout, weight decay, and early stopping. Includes hyperparameter tuning, unfrozen embeddings, and additional layers for improved performance.
- **Evaluation**: Accuracy, precision, recall, and F1-score on test set.

## Dataset

- **Source**: IMDB Dataset (50,000 movie reviews).
- **Columns**: `review` (text), `sentiment` (positive/negative).
- **File**: `IMDB Dataset.csv` (not included; downloaded from Kaggle).

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd sentiment_preprocessing
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the spaCy English model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. Download GloVe embeddings (for neural approach):
   ```python
   import gensim.downloader as api
   model = api.load("glove-wiki-gigaword-100")
   ```

## Usage

1. Place `IMDB Dataset.csv` in the project directory.

### Classical Approach
2. Open `classical-sentiment-preprocessing.ipynb` in Jupyter Notebook.
3. Run cells sequentially:
   - Load and preprocess data.
   - Vectorize text using BoW and TF-IDF.
   - Train and evaluate models (Logistic Regression, Naive Bayes).
   - Outputs include trained models and evaluation metrics.

### Neural Approach
2. Open `neural-sentiment-preprocessing.ipynb` in Jupyter Notebook.
3. Run cells sequentially:
   - Load and preprocess data.
   - Generate n-grams.
   - Build vocabulary and embedding matrix.
   - Convert to sequences and split data.
   - Perform hyperparameter tuning.
   - Train the neural model with early stopping.
   - Evaluate on test set.
   - Outputs include trained model (`best_tuned_model.pth`) and metrics.

### Example Output
- Vocabulary size: ~107k tokens (including specials).
- Sequences: Lists of integers representing words.
- Vectorized matrices for classical models.
- Neural model: Tuned classifier with accuracy, precision, recall.

## Results and Comparison

### Classical Models (Test Set Performance)
- **TF-IDF + Logistic Regression**: Accuracy 86%, Precision 87%/85%, Recall 85%/87%.
- **TF-IDF + Naive Bayes**: Accuracy 83%, Precision 84%/82%, Recall 81%/85%.
- **BoW + Naive Bayes**: Accuracy 83%, Precision 84%/82%, Recall 81%/84%.
- **BoW + Logistic Regression**: Accuracy 86%, Precision 87%/85%, Recall 85%/87%.

### Neural Model (Test Set Performance, After Tuning)
- **Accuracy**: 89.47%
- **Precision**: 87.71%
- **Recall**: 91.8%
- **F1-Score**: ~89.7% (estimated)

### Comparison
- **Strengths of Classical**: Simpler, faster training, strong baselines (83-86% accuracy) with BoW/TF-IDF capturing word frequencies effectively. Logistic Regression and Naive Bayes are robust for text classification without needing large data.
- **Strengths of Neural**: Better at capturing context and semantics with embeddings and layers. The tuned model (unfrozen embeddings, extra FC layers) outperforms classical methods (89.47% vs. 86%), demonstrating neural's potential with proper optimization.
- **Overall**: Classical methods provide excellent baselines with minimal setup; neural excels with tuning and offers better scalability for complex tasks. The neural approach achieved superior performance on this dataset.

## Dependencies

- pandas
- numpy
- scikit-learn
- spacy (with `en_core_web_sm` model)
- nltk
- torch
- gensim

## Project Structure

```
sentiment_preprocessing/
├── classical-sentiment-preprocessing.ipynb  # Classical ML approach with BoW/TF-IDF
├── neural-sentiment-preprocessing.ipynb     # Neural network data preparation and training
├── IMDB Dataset.csv                         # Dataset (user-provided)
├── requirements.txt                         # Python dependencies
├── README.md                                # This file
├── neural_preprocessing.pkl                 # Saved preprocessing artifacts (generated by neural notebook)
├── best_tuned_model.pth                     # Trained neural model (generated by neural notebook)
└── vectorizers.pkl                          # Saved vectorizers (generated by classical notebook)
```
