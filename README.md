# LSTM_SVMFusion
Hybrid Deep Learning Approach for Stock Price Prediction
# Stock Price Prediction using LSTM and SVM

This project demonstrates a hybrid approach to predict stock prices by combining the power of Long Short-Term Memory (LSTM) networks and Support Vector Machines (SVM). The aim of this project is to create a robust and accurate prediction model that takes advantage of both deep learning and traditional machine learning techniques.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

## Introduction

Stock price prediction has always been a challenging task due to its inherent volatility and sensitivity to various factors. This project explores the combination of LSTM, a type of recurrent neural network, and SVM, a well-established machine learning algorithm, to enhance the accuracy of stock price predictions.

## Dataset

The dataset used for this project contains historical stock price data. It includes features such as opening price, closing price, trading volume, and other relevant metrics. The data is preprocessed and split into training and testing sets to train and evaluate the prediction models.

## Methodology

1. **Data Preprocessing**: The raw stock price data is cleaned, normalized, and prepared for training. It's crucial to ensure the data is in a suitable format for both LSTM and SVM models.

2. **LSTM Model**: A Long Short-Term Memory network is employed to capture temporal patterns and dependencies in the stock price data. The LSTM model is trained using the training data to predict future stock prices.

3. **SVM Model**: A Support Vector Machine is used to build a predictive model based on historical stock price features. The SVM model aims to find a hyperplane that best separates different classes of data.

4. **Hybrid Fusion**: The predictions from both the LSTM and SVM models are combined using a fusion technique. This fusion leverages the strengths of both models to improve overall prediction accuracy.

## Installation

1. Clone this repository: `git clone https://github.com/772003pranav/stock-price-prediction.git`
2. Install the required dependencies: `pip install -r requirements.txt`

## Usage

1. Place your preprocessed dataset in the `data` directory.
2. Run the LSTM and SVM scripts to train the models: `python train_lstm.py` and `python train_svm.py`.
3. Run the fusion script to combine predictions: `python fuse_predictions.py`.
4. Evaluate the hybrid predictions and analyze the results.

## Results

The project's success is evaluated based on various metrics, including Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and accuracy of directional predictions. Comparative analysis of individual LSTM and SVM models with the hybrid approach is presented in the project documentation.

## Conclusion

This project showcases the potential of combining LSTM and SVM models to predict stock prices. It's important to note that stock market prediction is inherently uncertain, and this hybrid approach serves as a tool for informed decision-making rather than a guaranteed prediction tool.
