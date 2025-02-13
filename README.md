Spam Message Classifier

📌 Project Overview

This project is a Spam Message Classifier that detects whether a given text message is spam or ham (not spam). It uses Natural Language Processing (NLP) techniques and Machine Learning (ML) to classify messages efficiently.

🎯 Goal

The objective of this project is to build a reliable spam detection model using TF-IDF vectorization and a machine learning classifier. This model can be used in applications like email filtering, SMS classification, and chatbot security.

🛠 Tech Stack Used

Python (Programming Language)

Scikit-learn (ML Models & Evaluation)

Pandas & NumPy (Data Handling)

Matplotlib & Seaborn (Visualization)

NLTK (Text Preprocessing)

Joblib (Model Serialization)

📝 Dataset

The dataset used for this project is a widely available SMS Spam Collection dataset. It contains 5,574 messages labeled as ham (legitimate) or spam.

🔍 How It Works

Data Preprocessing

Removing stopwords and punctuations

Tokenization

Lemmatization

Feature Engineering

Converting text to numerical format using TF-IDF Vectorization

Model Training

Trained using a Naïve Bayes classifier (or alternative ML models like Logistic Regression)

Evaluation

Accuracy: 96.68%

Performance Metrics: Precision, Recall, F1-score

Confusion Matrix Visualization

Saving the Model

The trained model and vectorizer are saved as .pkl files for future predictions.

📊 Results & Accuracy

Accuracy: 96.68%
