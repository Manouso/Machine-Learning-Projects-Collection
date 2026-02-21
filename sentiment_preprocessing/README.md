# IMDB Sentiment Text Preprocessing

This project demonstrates text preprocessing techniques for sentiment analysis on the IMDB movie reviews dataset. It includes two approaches: classical machine learning methods using Bag-of-Words (BoW) and TF-IDF vectorization, and neural network preparation with vocabulary building and sequence conversion. The project also includes model training and evaluation for both approaches, with comparisons.

## Features

### Classical Approach (`classical-sentiment-preprocessing.ipynb`)
- **Text Preprocessing**: Lemmatization, stopword removal, punctuation filtering, HTML tag removal, lowercasing, and alphabetic token filtering using spaCy. Preprocessing is done after train/test split to prevent data leakage.
- **Vectorization**: BoW and TF-IDF with n-gram support (unigrams and bigrams), optimized for 5000 features.
- **Data Splitting**: Train/test split (80/20) with stratification for robust evaluation.
- **Model Training**: Logistic Regression and Naive Bayes classifiers with cross-validation.
- **Evaluation**: Classification reports, confusion matrices, and ROC AUC scores.
- **Saving**: All trained models and vectorizers saved for deployment.

### Neural Approach (`neural-sentiment-preprocessing.ipynb`)
- **Text Preprocessing**: Lemmatization, stopword removal, and punctuation filtering using spaCy.
- **N-gram Generation**: Create bigrams from cleaned text.
- **Vocabulary Building**: Construct a vocabulary with special tokens (`<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`) and word-to-index mappings.
- **Sequence Conversion**: Transform text into numerical sequences for neural model input.
- **Embedding Preparation**: Load GloVe embeddings and create an embedding matrix.
- **Model Training**: PyTorch-based neural classifier with embedding layer, fully connected layers, dropout, weight decay, and early stopping. Includes hyperparameter tuning based on validation loss minimization, unfrozen embeddings, and additional layers for improved performance.
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
   - Load and preprocess data with enhanced cleaning.
   - Split into train/val/test sets.
   - Vectorize text using BoW and TF-IDF.
   - Tune hyperparameters for Naive Bayes.
   - Train and evaluate models (Logistic Regression, Naive Bayes).
   - Outputs include trained models, vectorizers, and evaluation metrics.

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
- **TF-IDF + Logistic Regression**: Accuracy 89%, Precision 89%/88%, Recall 88%/90%, F1 89%/89%, ROC AUC 0.9544.
- **TF-IDF + Naive Bayes**: Accuracy 85%, Precision 87%/84%, Recall 83%/87%, F1 85%/85%, ROC AUC 0.9282.
- **BoW + Naive Bayes**: Accuracy 85%, Precision 85%/84%, Recall 84%/85%, F1 84%/85%, ROC AUC 0.9123.
- **BoW + Logistic Regression**: Accuracy 88%, Precision 89%/88%, Recall 87%/90%, F1 88%/89%, ROC AUC 0.9482.

### Neural Models (Test Set Performance, After Tuning on Validation Loss)
- **SemanticClassifier (Feedforward with Pooling)**: Accuracy 87.12%, Precision 83.11%, Recall 93.18%, F1 87.86%
- **SemanticClassifierGRU (Bidirectional GRU)**: Accuracy 87.42%, Precision 84.59%, Recall 91.52%, F1 87.92%
- **SemanticClassifierLSTM (Bidirectional LSTM)**: Accuracy 88.33%, Precision 89.16%, Recall 87.27%, F1 88.21%
- **AttwSemanticClassifierGRU (Attention-based GRU)**: Accuracy 88.33%, Precision 87.10%, Recall 90.00%, F1 88.52%
- **AttwSemanticClassifierLSTM (Attention-based LSTM)**: Accuracy 88.26%, Precision 85.82%, Recall 91.67%, F1 88.64%

### Comparison
- **Strengths of Classical**: Simpler, faster training, strong baselines (85-89% accuracy) with BoW/TF-IDF capturing word frequencies effectively. Logistic Regression provides robust, interpretable results with hyperparameter tuning. Prevents data leakage and uses proper validation. Slightly outperforms neural in F1-score (up to 0.89).
- **Strengths of Neural**: Better at capturing context and semantics with embeddings, RNNs, and attention mechanisms. Competitive performance (87-89% F1), with attention models offering interpretability. Scales well for larger datasets and complex patterns.
- **Overall**: Classical methods provide reliable, high-performance baselines with minimal complexity. Neural approaches are strong contenders, especially with attention for nuanced tasks, but classical edges out here on raw metrics.

### Why Classical Outperforms Neural Here
Despite neural networks' potential, the classical approach (especially TF-IDF + Logistic Regression) achieves slightly better F1 (0.89 vs. 0.8864) due to:
- **Dataset Suitability**: IMDB (50k samples) is sufficient for classical methods to learn effective linear patterns, but not massive enough for neural networks to fully leverage deep representations without overfitting.
- **Task Nature**: Sentiment analysis often relies on keyword presence and frequency (well-captured by TF-IDF), not requiring complex sequential modeling. Linear models excel at separable data.
- **Tuning and Overfitting**: Neural models showed signs of overfitting (e.g., increasing val loss in later epochs), while classical uses stable CV. The neural tuning was extensive and now optimized for validation loss minimization to improve generalization, but didn't surpass the baseline.
- **Efficiency and Interpretability**: Classical is faster to train/deploy and provides feature importance (e.g., which words drive predictions), making it practical for production.
- **Baseline Robustness**: For many NLP tasks, classical methods like Logistic Regression remain state-of-the-art unless using advanced transformers (e.g., BERT), which could push neural performance higher.

In summary, classical wins here because the problem is "solvable" with simpler methods—neural would likely dominate with larger, noisier data or more complex tasks.
## Future Improvements
- **Advanced Models**: Integrate transformer-based models like BERT or DistilBERT for potential 90%+ accuracy, leveraging pre-trained contextual embeddings.
- **Data Augmentation**: Apply synonym replacement, back-translation, or noise injection to increase dataset diversity and reduce overfitting.

- **Ensemble Methods**: Combine predictions from multiple models (e.g., feedforward + RNN) for improved robustness.
- **Sequence Length Handling**: Truncate or use attention mechanisms for very long reviews to manage memory and focus on key parts.

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
├── sentiment_models.pkl                     # Saved models and vectorizers (generated by classical notebook)
└── various .pth and .json files              # Additional neural model checkpoints
```
