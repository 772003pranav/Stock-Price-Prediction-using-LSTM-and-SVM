# Stock Price Prediction Readme

## Introduction

This repository contains code for predicting stock prices using two different machine learning models: Support Vector Machine (SVM) and Long Short-Term Memory (LSTM) neural networks. The objective of this project is to provide a practical example of how to approach stock price prediction using different techniques and models.

## Dataset

The dataset used for this project is crucial for the training and evaluation of our models. It is loaded from a CSV file named "TCS1.csv." This dataset contains historical stock price data, including the date and stock price values. Accurate and well-structured data is essential for building reliable predictive models.

## Support Vector Machine (SVM) Model

The SVM model is one of the approaches we use for stock price prediction. Here are the key steps involved:

1. **Data Preparation**: We begin by loading the dataset and preprocessing it. In particular, we convert the 'Date' column to datetime to ensure proper time series handling.

2. **Model Building**: We construct an SVM model with a linear kernel using Scikit-Learn's `SVR` class. The choice of kernel function can significantly impact the model's performance, and in this case, we use a linear kernel as a starting point.

3. **Model Training**: The SVM model is trained on the training data. The training phase is essential for the model to learn patterns and relationships within the data.

4. **Prediction**: After training, we utilize the trained SVM model to make predictions on the test data.

5. **Performance Evaluation**: To assess the model's accuracy, we calculate the Root Mean Squared Error (RMSE). RMSE is a widely used metric for regression tasks and helps quantify the prediction errors.

## Long Short-Term Memory (LSTM) Model

In addition to the SVM model, we also implement an LSTM neural network for stock price prediction. Here's how it works:

1. **Model Architecture**: We define an LSTM model using TensorFlow's Keras API. LSTMs are well-suited for time series data due to their ability to capture temporal dependencies.

2. **Model Compilation**: The LSTM model is compiled with the Adam optimizer and a mean squared error (MSE) loss function. These choices of optimizer and loss function are common for regression tasks.

3. **Model Training**: The LSTM model is trained on the training data. While the code in this repository trains the model for a single epoch, you can adjust the number of epochs to fine-tune the model's performance.

4. **Prediction**: After training, we use the LSTM model to make predictions on the test data.

5. **Data Imputation**: Prior to calculating the RMSE, we impute any missing values in the LSTM predictions to ensure a fair comparison with the SVM model.

6. **Performance Evaluation**: Similar to the SVM model, we calculate the RMSE to evaluate the accuracy of the LSTM model's predictions.

## RMSE Comparison

After training both models, the code calculates and prints the RMSE for both the SVM and LSTM models. This comparison provides insights into which model performs better for this specific stock price prediction task.

## Plotting Predictions

To visualize the model's predictions, we generate a plot that displays the true stock prices, LSTM predictions, and SVM predictions. This graphical representation helps us understand how well the models are capturing the underlying patterns in the data.

## Customization and Experimentation

Feel free to customize the code and dataset to work with your own stock price data or experiment with different machine learning models and hyperparameters. Stock price prediction is a complex task with many factors to consider, and this repository serves as a starting point for your own research and experiments.

We wish you the best of luck with your stock price prediction project!
