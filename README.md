# Fraud Detection

## General info

This project focuses on detecting anomalies in credit card transactions using machine learning techniques and autoencoders. The primary objective is to predict whether a given transaction is fraudulent or legitimate. The workflow includes comprehensive data analysis, preprocessing, and model development using algorithms such as Isolation Forest, Local Outlier Factor (LOF), One-Class Support Vector Machine (OneClassSVM), and deep learning-based autoencoder models.

### Dataset

The dataset contains transactions made by credit cards in two days in 2013 by European cardholders and contains frauds as well. It comes from Kaggle and can be find [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Motivation
The objective of this project is to predict whether a given credit card transaction is fraudulent. Credit card fraud can take various forms — from the use of stolen cards to more aggressive tactics such as account takeovers and counterfeiting. With the rapid growth of online transactions, the scale and complexity of such frauds continue to rise.

Anomaly detection, an unsupervised machine learning technique, plays a crucial role in identifying such fraudulent activities. It works by spotting unusual data points—called anomalies—that deviate significantly from typical patterns in the dataset. These anomalies often signal behavior that doesn't align with normal usage, making this technique particularly effective in identifying suspicious or fraudulent transactions.

## Project contains:
- fraud detection using machine learning models - **Fraud_detection.ipynb**
- fraud detection using autoencoder model - **Fraud_Autoencoder.ipynb**

## Summary
The primary goal of this project was to predict whether a given credit card transaction was fraudulent. The process involved extensive data analysis, data preprocessing, and the development of machine learning models using Isolation Forest, Local Outlier Factor (LOF), and One-Class Support Vector Machine (OneClassSVM).

In the first phase, I evaluated the performance of each model using various metrics, including accuracy, F1 score, recall, and the confusion matrix. Among these, One-Class SVM emerged as the most effective, achieving a recall of 84%. This indicates the model successfully identified the majority of fraudulent cases, minimizing the false negatives — which is critical in fraud detection scenarios.

In the second phase, I implemented an autoencoder-based approach for anomaly detection. The model was trained solely on normal (non-fraudulent) transactions, excluding any suspicious data. I assessed its performance using reconstruction error, recall, and the confusion matrix. The autoencoder was able to detect 83% of the anomalies based on recall, also maintaining a low false negative rate — ensuring that most fraud cases were successfully identified.

## Technologies

The project is created with:
- Python 3.9
- libraries: tensorflow, pandas, numpy, scikit-learn, seaborn, matplotlib.

**Running the project:**

To run this project use Jupyter Notebook or Google Colab.

### References:
- [Anomaly detection algorithms](https://towardsdatascience.com/5-anomaly-detection-algorithms-every-data-scientist-should-know-b36c3605ea16)
- [Credit card fraud detection using autoencoders](https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd)
- [Complete guide to anomaly detection with autoencoders](https://www.analyticsvidhya.com/blog/2022/01/complete-guide-to-anomaly-detection-with-autoencoders-using-tensorflow/)

