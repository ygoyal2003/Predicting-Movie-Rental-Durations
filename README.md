# Rental Duration Prediction Using Machine Learning

This project aims to predict the duration of rentals based on various features in a dataset. Multiple machine learning models are implemented and evaluated to find the best-performing model, including Lasso Regression, Linear Regression, and Random Forest.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Modeling and Evaluation](#modeling-and-evaluation)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project involves building models that predict how long a rental will last, using features from the dataset such as special features of the rented items. The models are evaluated based on their Mean Squared Error (MSE), and the Random Forest model is selected as the best-performing model.

## Features
- **Data Preprocessing**: 
  - Extracted rental duration from the difference between `rental_date` and `return_date`.
  - Created dummy variables for the presence of deleted scenes and behind-the-scenes features in the special features column.
  
- **Lasso Regression**: 
  - Used to perform feature selection by retaining only positive coefficients.

- **Linear Regression and Random Forest**: 
  - Linear regression is applied to the Lasso-selected features.
  - Random Forest with hyperparameter tuning via `RandomizedSearchCV` is used to optimize performance.

- **Model Evaluation**:
  - Models are evaluated using Mean Squared Error (MSE) on the test set, with Random Forest producing the lowest MSE.

## Technologies Used
- Python
- Pandas and NumPy for data manipulation
- Scikit-learn for machine learning models
- RandomizedSearchCV for hyperparameter tuning
- 
## Modeling and Evaluation
- **Lasso Regression**: Performs feature selection by eliminating features with zero coefficients.
- **Linear Regression**: Trained on the selected features from the Lasso model.
- **Random Forest**: Hyperparameters (`n_estimators`, `max_depth`) are optimized using `RandomizedSearchCV`. This model provided the lowest Mean Squared Error (MSE) and was selected as the best model.
