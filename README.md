# ChurnPrediction Using XGBoost

This project aims to predict customer churn for a bank using machine learning techniques. The model has been trained to classify whether a customer will churn (leave the bank) or not, based on various features like credit score, age, gender, balance, and more.

## Overview

Customer churn prediction is an important task for businesses as it helps in identifying customers who are likely to leave the service, allowing for proactive retention strategies. This project utilizes the XGBoost algorithm, a powerful and efficient implementation of gradient boosting, to build a predictive model for customer churn.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Comparison](#comparison)
- [Model Training](#model-training)
- [Algorithm Explanation](#algorithm-explanation)
- [Results](#results)

## Installation

To run this project, you need to install the required packages listed in `requirements.txt`. Use the following command to install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository.
2. Ensure you have the dataset `Churn_Modelling.csv` in the project directory.
3. Run the `xgb.py` script to train the model.
4. Use `app.py` to serve the model and make predictions via a web interface.


## Comparison

**Model Comparison: Logistic Regression, XGBoost, Random Forest, KNN, Naive Bayes**
In the 'ChurnPrediction.ipynb' notebook, I compared the performance of five different machine learning algorithms for predicting customer churn. The algorithms under comparison are:

1. **Logistic Regression**
2. **XGBoost**
3. **Random Forest**
4. **K-Nearest Neighbors (KNN)**
5. **Naive Bayes**


## Model Training

The model training process is outlined in the `xgb.py` script. Below are the main steps involved:

1. **Data Loading**: Load the customer data from the CSV file.
2. **Data Preprocessing**: Encode categorical variables, drop unnecessary columns, and split the data into features and target variable.
3. **Data Balancing**: Use SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance in the training data.
4. **Model Training**: Train an XGBoost classifier on the balanced training data.
5. **Model Evaluation**: Evaluate the model on both training and test datasets to check for performance metrics.

## Algorithm Explanation

### XGBoost (Extreme Gradient Boosting)

XGBoost is an implementation of gradient boosted decision trees designed for speed and performance. It is a scalable and efficient algorithm that works well with large datasets and provides state-of-the-art results on many machine learning problems. Here's a brief explanation of how it works:

- **Decision Trees**: XGBoost builds an ensemble of decision trees, where each tree is trained to correct the errors of the previous ones.
- **Gradient Boosting**: Instead of assigning equal weights to all predictions, XGBoost uses gradient descent to minimize the loss function by adjusting weights. Each tree is added sequentially, and new trees correct the errors made by the existing ensemble.
- **Regularization**: XGBoost includes regularization terms in its objective function, which helps to control overfitting and improve model generalization.
- **Parallel Processing**: XGBoost leverages parallel processing for faster computation, making it highly efficient.

## Results

The model's performance is evaluated using classification metrics such as precision, recall, F1-score, and accuracy. Below are the results for the training and test datasets:

### Training Data Classification Report

![image](https://github.com/user-attachments/assets/c9e6438f-063b-4514-8eba-6108c6e29ddd)


![image](https://github.com/user-attachments/assets/59d4ddc1-a995-4f79-aa3d-5899614de08a)


### Final Output

https://github.com/user-attachments/assets/65b67375-45d5-46d1-bec8-a701df30d973



