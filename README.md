# Predicting Student Performance from Game Play

## Summary
This repository explores predicting student correctness in an educational game ("Jo Wilder") using the Kaggle challenge dataset: Predict Student Performance.

---

## Overview

### Definition of the Task
The Kaggle challenge involves predicting the correctness of a student's response in an educational game based on their interaction data, such as room coordinates, event types, and time spent. This task is framed as a binary classification problem.

---

## Approach

My approach focuses on:
- Preprocessing the dataset by handling missing values, encoding categorical variables, and scaling numerical features.
- Engineering session-level features (e.g., aggregated statistics).
- Employing a Random Forest Classifier to predict student performance.
- Analyzing feature importance and visualizing interaction patterns to guide feature engineering.

---

## Summary of Performance Achieved
The model achieved an accuracy of 1.0 on the test set, which suggests overfitting. Future improvements include addressing class imbalance and exploring advanced machine learning models.

---

## Summary of Work Done

### Data Type:
- **Input:** A CSV file containing user interactions in a game (e.g., session ID, room coordinates, hover durations).
- **Output:** Binary target variable (`correct`), indicating whether the student answered correctly.
- **Size:** ~10,000 data points.

### Instances:
- **Training:** 6,000 samples.
- **Test:** 2,000 samples.
- **Validation:** Not explicitly defined; a portion of the test set was used for this purpose.

### Preprocessing / Cleanup:
- **Missing Values:** Imputed missing values in numerical columns with the median.
- **Categorical Encoding:** Used one-hot and frequency encoding for features like `event_name` and `fqid`.
- **Outlier Handling:** Capped outliers in `hover_duration` and `elapsed_time` using the interquartile range (IQR) method.
- **Feature Scaling:** Standardized numerical features for consistency.

---

## Data Visualization

- **Class Distribution:** Revealed significant imbalance, with most responses being incorrect.
- **Feature Distributions:** Histograms showed clustering in features like `room_coor_x` and `hover_duration`.
- **Feature-Class Relationship:** Notable differences in distributions for features like `elapsed_time` and `room_coor_x` between correct and incorrect responses.

---

## Problem Formulation

- **Input/Output:**
  - **Input:** Preprocessed interaction data.
  - **Output:** Binary classification (correct or incorrect).
- **Model:** Random Forest Classifier for its robustness and handling of mixed data types.
- **Hyperparameters:** Default settings, with plans for optimization in future iterations.

---

## Training

- **Process:** Used Python libraries like pandas, scikit-learn, and matplotlib. Training data split into 80% training and 20% testing.
- **Duration:** Approximately 10-15 minutes.
- **Stopping Criteria:** Training stopped after achieving high test accuracy; early stopping was not implemented.
- **Challenges:** Addressing non-numeric values in test data required repeated encoding steps.

---
## Feature Engineering

1. **Library Imports:**
   - Used `pandas` for data handling and `matplotlib.pyplot` for creating visualizations, such as histograms.

2. **Identifying Numeric Columns:**
   - Focused on numeric data (`int64` and `float64`) for statistical analysis, including the calculation of mean and standard deviation.

3. **Statistical Analysis:**
   - Computed the **mean** to understand the central tendency and the **standard deviation** to measure data variability using `pandas` functions (`mean()` and `std()`).

4. **Histogram Visualizations:**
   - Created histograms for numeric columns to observe distributions and detect patterns or outliers. Used `matplotlib` for plotting, ensuring NaN values were removed for accuracy.

5. **Outlier Handling:**
   - Identified and capped outliers using the Interquartile Range (IQR) method for features such as `hover_duration` and `elapsed_time`.

6. **Aggregated Session-Level Features:**
   - Derived features summarizing interaction data, including averages, totals, and counts for each session.
  
### Insights Gained:
- Histograms highlighted clustering in gameplay data, suggesting patterns.
- Engineered features (e.g., elapsed time per session, interaction count) provided additional predictive power for classification.
---

## Performance Comparison

- **Metrics:**
  - Primary: Accuracy.
  - Secondary: Precision and recall (to be explored for class imbalance).
- **Results:** Random Forest achieved an accuracy of 1.0, likely overfitted.

---

## Conclusions

- **Strengths:** The model successfully captured relationships between features and the target variable.
- **Weaknesses:** Overfitting suggests a need for better generalization techniques.
- **Future Improvements:**
  - Address class imbalance using SMOTE or weighted loss functions.
  - Experiment with advanced models like XGBoost, LightGBM, or Neural Networks.
  - Incorporate session-level sequential data for richer features.
  - Implement k-fold cross-validation.

---

## How to Reproduce Results

1. **Setup Instructions:**
   - Install necessary libraries: `pip install pandas scikit-learn matplotlib`.
   - Download the dataset from Kaggle and place it in the `/content/train/` directory.
2. **Training:**
   - Run the provided notebook file, ensuring proper handling of missing columns during test preprocessing.
3. **Evaluation:**
   - Use the model to predict the test dataset and compare predictions to ground truth.

---

## Overview of Files in Repository

- `utils.py`: Functions for preprocessing (e.g., handling missing values, encoding).
- `preprocess.ipynb`: Handles data cleaning and feature engineering.
- `models.py`: Contains Random Forest implementation and other ML models.
- `training-model.ipynb`: Trains the Random Forest model and generates predictions for evaluation.
- `performance.ipynb`: Evaluates the trained model and generates a Kaggle submission file.

---

## Software Setup

- **Required Packages:** `pandas`, `scikit-learn`, `matplotlib`, `seaborn`.
- **Installation Instructions:** Install using `pip install <package-name>`.
- **Data Access:** Dataset is available on the Kaggle challenge page.

---

## Citations

- Kaggle: Predict Student Performance from Game Play.
