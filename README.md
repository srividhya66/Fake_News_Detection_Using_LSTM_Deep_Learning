🧠 Fake News Detection using LSTM | Deep Learning NLP Project
📘 Project Overview

This project aims to detect fake news articles using Natural Language Processing (NLP) and Deep Learning techniques.
We build and train a Long Short-Term Memory (LSTM) based model to classify news articles as “Fake” or “Real” based on their textual content.

The model learns semantic patterns, word dependencies, and contextual meaning within news data to effectively distinguish misleading information from genuine content.

🚀 Features

📰 Text Classification using an LSTM neural network

🧹 Data Cleaning & Preprocessing (stopword removal, tokenization, padding, etc.)

🔤 Word Embeddings using GloVe or Keras Embedding layer

📊 Model Training & Evaluation with accuracy and loss visualizations

⚙️ Binary Classification Output: Fake (0) / Real (1)

🧩 Model Architecture

Input Layer: Preprocessed and tokenized text sequences
Embedding Layer: Converts words into dense vector representations
LSTM Layer: Captures temporal and contextual dependencies in text
Dense Layers: Perform nonlinear transformations for classification
Output Layer: Sigmoid activation for binary classification

🧠 Technologies Used

Python 3.x

TensorFlow / Keras

NumPy

Pandas

Matplotlib / Seaborn

NLTK / re (Regex) for text preprocessing

📁 Dataset

You can use publicly available datasets such as:

Fake News Dataset – Kaggle

LIAR Dataset – Fake News Detection

Dataset columns typically include:

title – Headline of the news article

text – Main body of the article

label – Target label (Fake / Real)

⚙️ How to Run

Clone this repository

git clone https://github.com/<your-username>/FakeNewsDetection_LSTM.git
cd FakeNewsDetection_LSTM


Install dependencies

pip install -r requirements.txt


Download GloVe embeddings (optional)

wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip


Run the notebook

jupyter notebook Fake_News_Detection_Using_LSTM.ipynb

📈 Results

Achieved ~90% accuracy on validation data (depending on dataset)

Visualized training vs. validation loss and accuracy

Model successfully identifies most fake news articles based on content patterns

Example output:

Input:  "Breaking: Scientists discover chocolate cures stress!"
Prediction:  Fake 🟥

💡 Applications

Automated fact-checking systems

News content filtering and moderation

Social media misinformation detection

Educational tools for media literacy

🧩 Future Work

Integrate BiLSTM or Transformer-based models (BERT) for better context understanding

Include headline + body matching for improved accuracy

Build a web-based interface using Flask or Streamlit for real-time detection
