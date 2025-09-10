# 📂 Natural Language Processing (NLP) Projects

This repository contains projects and homework assignments focused on **text data processing, embeddings, and deep learning models** for sentiment analysis and classification.

## 📑 Projects Index

| Project | Tagline | Link |
|---------|---------|------|
| Project 1 | Sentiment Analysis with Amazon Reviews (Traditional ML) | [View](#-project-1--sentiment-analysis-with-amazon-reviews-traditional-ml) |
| Project 2 | Word Embeddings & Deep Learning for Sentiment Analysis | [View](#-project-2--word-embeddings--deep-learning-for-sentiment-analysis) |
| Project 3 | HMM-Based Part-of-Speech Tagging | [View](#-project-3--hmm-based-part-of-speech-tagging) |
| Project 4 | Named Entity Recognition with BLSTM and GloVe | [View](#-project-4--named-entity-recognition-with-blstm-and-glove) |


---

## 📌 Project 1 — Sentiment Analysis with Amazon Reviews (Traditional ML)

> **One-liner:** Building sentiment classifiers using text preprocessing, TF-IDF features, and traditional machine learning models.  

This project focuses on **sentiment analysis using Amazon product reviews**.  
The main steps include:

1. **Dataset Preparation** – Selecting positive and negative reviews, creating binary sentiment labels, and splitting into training/testing sets.  
2. **Data Cleaning** – Removing noise such as HTML tags, URLs, non-alphabetical characters, extra spaces, and handling contractions.  
3. **Preprocessing** – Tokenization, stopword removal, and lemmatization using NLTK.  
4. **Feature Extraction** – Transforming text into numerical representations with TF-IDF.  
5. **Model Training & Evaluation** – Training and evaluating multiple classifiers:
   - Perceptron  
   - Support Vector Machine (SVM)  
   - Logistic Regression  
   - Multinomial Naive Bayes  

The models are evaluated on **Accuracy, Precision, Recall, and F1-score** for both training and testing datasets.  

The repository includes a Jupyter Notebook with code, explanations, and results for all experiments.

---

## 📌 Project 2 — Word Embeddings & Deep Learning for Sentiment Analysis

> **One-liner:** Extending sentiment analysis with Word2Vec embeddings, feedforward networks, and CNNs for richer text representation.  

This project extends the first sentiment analysis task by introducing **word embeddings and neural models**.  
The main steps include:

1. **Dataset Generation** – Building a balanced dataset (250k reviews) with ternary sentiment labels: positive, negative, and neutral.  
2. **Word Embeddings** –  
   - Using pretrained **Google News Word2Vec** embeddings.  
   - Training a custom **Word2Vec** model on the dataset.  
   - Comparing semantic similarity examples (e.g., *King - Man + Woman = Queen*).  
3. **Traditional Models with Embeddings** –  
   - Averaging Word2Vec vectors per review and training **Perceptron** and **SVM** classifiers.  
   - Comparing performances using TF-IDF, pretrained embeddings, and custom embeddings.  
4. **Feedforward Neural Networks (FNN/MLP)** –  
   - Binary and ternary sentiment classification using averaged embeddings.  
   - Concatenation of first 10 word vectors per review for richer feature representations.  
5. **Convolutional Neural Networks (CNN)** –  
   - Training CNNs for sentiment analysis with padded/truncated review sequences.  
   - Binary and ternary classification settings.  

In total, models are evaluated across **16 different experimental setups**, comparing TF-IDF, pretrained Word2Vec, and custom Word2Vec embeddings with both traditional ML and deep learning models.  

The repository includes a Jupyter Notebook with code, explanations, and results for all experiments.

---

## 📌 Project 3 — HMM-Based Part-of-Speech Tagging  

> **One-liner:** Building an HMM model for POS tagging with vocabulary creation, parameter estimation, and decoding algorithms.  

This project applies **Hidden Markov Models (HMMs)** to the task of **part-of-speech tagging** using the **Wall Street Journal portion of the Penn Treebank** dataset.  
The main steps include:  

1. **Vocabulary Creation** –  
   - Construct a vocabulary from the training set.  
   - Replace rare words (frequency below a chosen threshold) with the special token `<unk>`.  
   - Save the vocabulary in `vocab.txt` with word, index, and frequency, sorted by frequency.  
   - Analyze vocabulary size and `<unk>` token frequency.  

2. **Model Learning** –  
   - Estimate **transition probabilities**:  
     \[
     t(s'|s) = \frac{\text{count}(s \to s')}{\text{count}(s)}
     \]  
   - Estimate **emission probabilities**:  
     \[
     e(x|s) = \frac{\text{count}(s \to x)}{\text{count}(s)}
     \]  
   - Save learned parameters to `hmm.json` (two dictionaries: `transition`, `emission`).  
   - Report the total number of parameters learned.  

3. **Greedy Decoding** –  
   - Implement greedy decoding for POS tagging.  
   - Evaluate tagging accuracy on the **development set**.  
   - Predict POS tags for the test set and save results in `greedy.out`.  

4. **Viterbi Decoding** –  
   - Implement the **Viterbi algorithm** for sequence decoding.  
   - Evaluate tagging accuracy on the **development set**.  
   - Predict POS tags for the test set and save results in `viterbi.out`.  

### 📂 Deliverables  
- `vocab.txt` → Vocabulary file with `<unk>` handling.  
- `hmm.json` → Learned emission and transition parameters.  
- `greedy.out` → POS predictions from greedy decoding.  
- `viterbi.out` → POS predictions from Viterbi decoding.    

The repository includes a Jupyter Notebook with code, explanations, and results for all experiments.

---

## 📌 Project 4 — Named Entity Recognition with BLSTM and GloVe  

> **One-liner:** Building a neural network for NER using bidirectional LSTM models and GloVe word embeddings, with an optional character-level CNN module for improved performance.  

This project focuses on **named entity recognition (NER)** using the **CoNLL-2003 corpus**. The task is to identify named entities in sentences, such as persons, organizations, locations, and miscellaneous entities.  

The main steps include:  

1. **Simple Bidirectional LSTM (BLSTM) Model** –  
   - Implement a BLSTM network with PyTorch: `Embedding → BLSTM → Linear → ELU → classifier`.  
   - Hyperparameters include embedding dimension (100), LSTM hidden size (256), LSTM dropout (0.33), and linear output dimension (128).  
   - Train the model on the training set using SGD, tune learning rate and batch size.  
   - Evaluate precision, recall, and F1 score on the development set.  

2. **BLSTM with GloVe Word Embeddings** –  
   - Initialize the embedding layer with **pretrained GloVe vectors**.  
   - Handle case-sensitivity conflicts between GloVe and NER data.  
   - Train and evaluate the model on dev data to improve performance.  

3. **Bonus: LSTM-CNN Model** –  
   - Extend the BLSTM model with a **CNN module** for character-level information.  
   - Character embedding dimension set to 30; tune kernel size, output dimensions, and number of CNN layers.  
   - Predict NER tags on dev and test data; evaluate improvements in F1 score.  

### 📂 Deliverables  
- `blstm1.pt` → Trained BLSTM model from Task 1.  
- `blstm2.pt` → Trained BLSTM model using GloVe embeddings from Task 2.  
- Prediction files: `dev1.out`, `dev2.out`, `test1.out`, `test2.out`.    

The repository includes a Jupyter Notebook with code, explanations, and results for all experiments.


