# Spam Email Classifier

This project implements machine learning models to classify emails as spam or non-spam (ham). It explores the use of Random Forest and Gradient Boosting classifiers to achieve this task.

## Overview

Spam email classification is a common problem in natural language processing and machine learning. The goal is to develop a model that can accurately distinguish between spam and non-spam emails based on their content.

## Dataset

The project uses the Email SMS Spam Collection Dataset. The dataset contains SMS messages labeled as spam or ham.

## Data Processing

The following data processing steps are performed:

- Removal of punctuation and special characters
- Conversion of text to lowercase
- Tokenization of text into individual words
- Removal of stopwords
- Stemming of words using the Porter stemmer algorithm

## Vectorization

The text data is vectorized using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization, which converts text documents into numerical vectors. This process transforms the textual content into a format suitable for machine learning models.

## Machine Learning Algorithm

Two machine learning algorithms are explored in this project:

1. Random Forest Classifier: A decision tree-based ensemble learning method that builds multiple decision trees and combines their predictions to improve accuracy.

2. Gradient Boosting Classifier: A machine learning technique that builds an ensemble of weak learners (typically decision trees) in a sequential manner, with each new model correcting errors made by the previous ones.

## Hyperparameter Tuning

Hyperparameters of the machine learning models are tuned using GridSearchCV, an exhaustive search technique that evaluates model performance across various hyperparameter combinations to find the optimal settings.

## Project Structure

The project consists of the following files:

- `05_Random_Forest_Classifier_Through_Cross_Validation.ipynb`: Jupyter Notebook containing code for building a basic Random Forest model with cross-validation.
- `05_Random_Forest_Classifier_Through_Holdout_Test_Methods.ipynb`: Jupyter Notebook containing code for building a Random Forest model with holdout test methods.
- `05_Random_Forest_Classifer_Evaluation_With_GridSearchCV.ipynb`: Jupyter Notebook containing code for evaluating Random Forest with GridSearchCV.
- `05_Gradient_Boosting_With_Grid_Search.ipynb`: Jupyter Notebook exploring Gradient Boosting model with grid-search.
- `05_Gradient_Boosting_With_GridSearchCV.ipynb`: Jupyter Notebook evaluating Gradient Boosting with GridSearchCV.
- `06_Final.ipynb`: Jupyter Notebook containing final evaluation of models.
- `SMSSpamCollection.tsv`: Dataset file containing SMS messages labeled as spam or ham.
- `README.md`: This file.

## Conclusion

Based on the evaluation results, the Random Forest model outperforms the Gradient Boosting model in terms of precision, accuracy, and computational efficiency. Both models achieve the same recall score. The README file provides insights into the data processing steps, feature engineering, model selection, hyperparameter tuning, and conclusions drawn from the project.

## Dependencies

- Python 3.x
- Jupyter Notebook
- scikit-learn
- pandas
- nltk
- matplotlib
