# IMDB Sentiment Text Preprocessing

This project demonstrates text preprocessing techniques for sentiment analysis on the IMDB movie reviews dataset. It includes text cleaning, n-gram generation, vocabulary building with special tokens, and conversion to numerical sequences for models (classical or neural).

## Features

- **Text Preprocessing**: Lemmatization, stopword removal, and punctuation filtering using spacy.
- **N-gram Generation**: Create bigrams (or other n-grams) from cleaned text.
- **Vocabulary Building**: Construct a vocabulary with special tokens (`<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`) and word-to-index mappings.
- **Sequence Conversion**: Transform text into numerical sequences (vectors) for model input.
- **Data Saving**: Export vocabulary and sequences to pickle files for reuse.

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

## Usage

1. Place `IMDB Dataset.csv` in the project directory.
2. Open `imdb_text_preprocessing.ipynb` in Jupyter Notebook.
3. Run cells sequentially:
   - Load and preprocess data.
   - Generate n-grams.
   - Build vocabulary.
   - Convert to sequences.
   - Save outputs (`vocabulary.pkl`, `sequences.pkl`).

### Example Output
- Vocabulary size: ~107k tokens (including specials).
- Sequences: Lists of integers representing words.

## Dependencies

- pandas
- numpy
- scikit-learn
- spacy (with `en_core_web_sm` model)
- nltk (for n-grams)

## Project Structure

```
sentiment_preprocessing/
├── classical_text_representation.ipynb  # Main notebook
├── IMDB Dataset.csv                     # Dataset (user-provided)
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
├── vocabulary.pkl                       # Saved vocabulary (generated)
└── sequences.pkl                        # Saved sequences (generated)
```
