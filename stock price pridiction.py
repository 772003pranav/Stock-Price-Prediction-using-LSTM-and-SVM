import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the stock price dataset into a pandas DataFrame
data = pd.read_csv('/content/TCS1.csv')
data['Date'] = pd.to_datetime(data['Date'])


# Build and train an SVM model
svm_model = SVR(kernel='linear')
svm_model.fit(X_train_imputed, y_train_imputed.ravel())

# Predict using the SVM model
svm_predictions = svm_model.predict(X_test_imputed)

# Reshape the svm_predictions array to be two-dimensional
svm_predictions_reshaped = svm_predictions.reshape(-1, 1)

# Define the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_imputed.shape[1], 1)))
lstm_model.add(LSTM(units=50, return_sequences=False))
lstm_model.add(Dense(units=25))
lstm_model.add(Dense(units=1))

# Compile the model
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
lstm_model.fit(X_train_imputed, y_train_imputed, batch_size=1, epochs=1)

# Predict using the LSTM model
lstm_predictions = lstm_model.predict(X_test_imputed)

# Impute missing values for lstm_predictions
lstm_predictions_imputed = imputer_y_test.transform(lstm_predictions)

# Calculate RMSE for both models
lstm_rmse = np.sqrt(mean_squared_error(y_test_imputed, lstm_predictions_imputed))
svm_rmse = np.sqrt(mean_squared_error(y_test_imputed, svm_predictions_imputed))

print(f"LSTM RMSE: {lstm_rmse}")
print(f"SVM RMSE: {svm_rmse}")

# Plot the original stock prices, LSTM predictions, and SVM predictions
plt.figure(figsize=(12, 6))
plt.plot(data['Date'][-len(y_test_imputed):], y_test_imputed, label='True Stock Price', color='blue')
plt.plot(data['Date'][-len(y_test_imputed):], lstm_predictions_imputed, label='LSTM Predictions', color='green')
plt.plot(data['Date'][-len(y_test_imputed):], svm_predictions_imputed, label='SVM Predictions', color='red')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



