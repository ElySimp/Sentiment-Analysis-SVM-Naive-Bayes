# Sentiment Analysis with Explainable AI (XAI): LinearSVC vs. Naïve Bayes

A comparative Natural Language Processing (NLP) benchmark evaluating **Linear Support Vector Machine (`LinearSVC`)** against **Multinomial Naïve Bayes (`MultinomialNB`)** on 50,000 IMDB movie reviews, paired with **LIME** local interpretability and global linear feature inspection.

## Dataset & Preprocessing Pipeline

- **Dataset:** IMDB Dataset of 50,000 Movie Reviews (balanced binary classification: `positive` / `negative`).
- **Split:** Stratified 80/20 split (`40,000` training / `10,000` test reviews, `random_state=42`).
- **Preprocessing (`svm-naive-bayes-xai.ipynb`):**
  1. Regex removal of HTML tags and entities.
  2. Lowercasing and NLTK word tokenization (`nltk.word_tokenize`).
  3. Filtering short tokens (`len <= 2`) and removing `224` domain-specific noise words.
  4. POS-tagged lemmatization (`WordNetLemmatizer` + `pos_tag`).
  5. N-gram vectorization via `TfidfVectorizer(ngram_range=(1, 2), stop_words="english")` producing **2,663,697** sparse features (`99.99%` sparsity).

## Model Evaluation Results (10,000 Held-Out Test Reviews)

| Model                                           |     Accuracy     |    Precision    |      Recall      |     F1-Score     |
| :---------------------------------------------- | :--------------: | :--------------: | :--------------: | :--------------: |
| **Multinomial Naïve Bayes**              |      87.89%      |      0.8848      |      0.8712      |      0.8780      |
| **LinearSVC (`GridSearchCV`, `C=1`)** | **90.41%** | **0.8930** | **0.9182** | **0.9054** |

## Explainable AI (XAI) Methodology

- **Local Explanations (LIME):** Wrapped `LinearSVC.decision_function` with a sigmoid calibration layer (`scipy.special.expit`) inside `LimeTextExplainer` to inspect top word contributions per review (`num_features=10`) and aggregate explanations across sample batches (highlighting high-impact positive tokens such as `excellent` and negative tokens such as `bad`).
- **Global Interpretability:** Inspected `LinearSVC.coef_[0]` weights across `984,247` positive-weight and `866,623` negative-weight n-gram features.

## Tech Stack

- Python, Scikit-Learn (`LinearSVC`, `MultinomialNB`, `TfidfVectorizer`, `GridSearchCV`), NLTK, LIME, SciPy, Pandas, NumPy

## Running the Notebook

1. Ensure `IMDB Dataset.csv` is accessible in your working or Kaggle input path.
2. Install dependencies (`scikit-learn`, `nltk`, `lime`, `scipy`, `pandas`, `numpy`) and open `svm-naive-bayes-xai.ipynb`.
