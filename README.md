Project Title

Predicting Student Performance from Game Play

One Sentence Summary

This repository explores predicting student correctness in an educational game ("Jo Wilder") using the Kaggle challenge dataset: Predict Student Performance.

Overview

Definition of the Task
The Kaggle challenge involves predicting the correctness of a student's response in an educational game based on their interaction data, such as room coordinates, event types, and time spent. This task is framed as a binary classification problem.

Approach

Our approach focuses on preprocessing the dataset by handling missing values, encoding categorical variables, and scaling numerical features. We engineered session-level features (e.g., aggregated statistics) and employed a Random Forest Classifier to predict student performance. We also analyzed feature importance and visualized interaction patterns to guide our feature engineering.

Summary of Performance Achieved

The model achieved an accuracy of 1.0 on the test set, but this suggests overfitting. Future improvements include addressing class imbalance and exploring advanced machine learning models.
Summary of Work Done

Data
Type:
Input: A CSV file containing user interactions in a game (e.g., session ID, room coordinates, hover durations).
Output: Binary target variable (correct), indicating whether the student answered correctly.
Size: ~10,000 data points.
Instances:
Training: 6,000 samples.
Test: 2,000 samples.
Validation: Not explicitly defined; a portion of the test set was used for this purpose.
Preprocessing / Cleanup
Missing values in numerical columns were imputed with the median.
Categorical features (e.g., event_name, fqid) were encoded using one-hot and frequency encoding.
Outliers in features like hover_duration and elapsed_time were identified and capped using the interquartile range (IQR) method.
Numerical features were standardized for consistency.
Data Visualization
Class Distribution:
Showed a significant imbalance, with most responses being incorrect.
Feature Distributions:
Histograms for features like room_coor_x and hover_duration revealed clustering around specific values, indicating patterns in gameplay behavior.
Feature-Class Relationship:
Features like elapsed_time and room_coor_x showed distinguishable distributions for correct vs. incorrect responses.
Problem Formulation
Input/Output:
Input: Preprocessed interaction data.
Output: Binary classification (correct or incorrect).
Model:
Random Forest Classifier was chosen for its ability to handle mixed data types and robustness to overfitting.
Hyperparameters: Default settings, with plans for optimization in future iterations.
Training
Process:
Training was conducted using Python libraries such as pandas, scikit-learn, and matplotlib.
Training data was split into 80% train and 20% test subsets.
Duration: Model training took approximately 10-15 minutes.
Stopping Criteria: Training stopped after achieving high accuracy on the test set. Early stopping was not implemented.
Challenges:
Handling non-numeric values in the test data required repeated encoding steps.
Performance Comparison
Metrics:
Accuracy (primary metric).
Precision and recall were not extensively analyzed but are critical due to class imbalance.
Results:
The Random Forest achieved an accuracy of 1.0, likely due to overfitting.
Conclusions
The model successfully identified relationships between features and the target variable, achieving high accuracy.
Overfitting indicates the need for better generalization techniques, such as cross-validation and regularization.
Future Work
Address Class Imbalance:
Apply SMOTE or weighted loss functions.
Experiment with Advanced Models:
Test Gradient Boosting Machines (e.g., XGBoost, LightGBM) or Neural Networks.
Feature Engineering:
Incorporate session-level sequential data (e.g., event order).
Robust Validation:
Implement k-fold cross-validation and stratified sampling.
How to Reproduce Results
Setup Instructions:
Install necessary libraries: pip install pandas scikit-learn matplotlib.
Download the dataset from Kaggle and place it in the /content/train/ directory.
Training:
Run the provided notebook file. Ensure that missing columns are added during test preprocessing.
Evaluation:
Use the model to predict the test dataset and compare predictions to ground truth.
Overview of Files in Repository
utils.py:
Functions for preprocessing (e.g., handling missing values, encoding).
preprocess.ipynb:
Handles data cleaning and feature engineering.
models.py:
Contains Random Forest implementation and other machine learning models.
training-model.ipynb:
Trains the Random Forest model and generates predictions for evaluation.
performance.ipynb:
Evaluates the trained model and generates a Kaggle submission file.
Software Setup
Required Packages:
pandas, scikit-learn, matplotlib, seaborn.
Installation Instructions:
Install using pip install <package-name>.
Data Access:
Dataset is available on the Kaggle challenge page.
Citations
Kaggle: Predict Student Performance from Game Play.
