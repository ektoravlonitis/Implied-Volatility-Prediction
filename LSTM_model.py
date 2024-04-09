# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 02:34:41 2024

@author: he_98
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Setting the font sizes
plt.rcParams['axes.titlesize'] = 20  # Title font size
plt.rcParams['axes.labelsize'] = 18  # Axis label font size
plt.rcParams['xtick.labelsize'] = 16  # X-axis tick label size
plt.rcParams['ytick.labelsize'] = 16  # Y-axis tick label size
plt.rcParams['legend.fontsize'] = 16  # Legend font size


# Loading the dataset
csv_file_path = 'main_columns_filtered.csv'
df = pd.read_csv(csv_file_path)

# Pivoting the DataFrame to have 'Strike Price' values as columns for 'Implied Volatility'
pivot_df = df.pivot_table(index=['Exchange Date', 'Days to Maturity', 'Close_Index'], 
                          columns='Strike Price', 
                          values='Implied Volatility', 
                          aggfunc='first').reset_index()

pivot_df.columns = [f'IV_{col}' if pd.api.types.is_number(col) else col for col in pivot_df.columns]

pivot_df.to_csv('pivot_csv.csv', index = False)

# Loading the combined_data.csv
comb_df = pd.read_csv('combined_data.csv')

# Unique strike prices from comb.csv
unique_strike_prices = comb_df['Strike Price'].unique()

# For each unique strike price, we filter comb_df and update pivot_df accordingly
for strike_price in unique_strike_prices:
    filtered_comb_df = comb_df[comb_df['Strike Price'] == strike_price]
    merged_df = pivot_df.merge(filtered_comb_df, how='left', on='Exchange Date')
    
    # Dropping the redundant 'Implied Volatility' column from the merged DataFrame
    merged_df = merged_df.drop(columns=['Implied Volatility'])

    for col in ['Days to Maturity', 'Close_Index']:
        if f'{col}_x' in merged_df.columns and f'{col}_y' in merged_df.columns:
            merged_df = merged_df.drop(columns=[f'{col}_y'])
            merged_df = merged_df.rename(columns={f'{col}_x': col})

    # Saving the new pivot table for this specific strike price
    merged_df.to_csv(f'pivot_df_{strike_price}.csv', index=False)


# Defining the number of steps
n_steps = 5

# Function to create sequences for LSTM
def create_sequences(data, target_col_index, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i, :])
        y.append(data[i, target_col_index])
    return np.array(X), np.array(y)

# Function to create non-overlapping sequences for LSTM
def create_non_overlapping_sequences(data, target_col_index, n_steps):
    X, y = [], []
    for i in range(0, len(data) - n_steps + 1, n_steps):
        X.append(data[i:i+n_steps, :])
        # The target is right after the last step of each sequence
        if i + n_steps < len(data):
            y.append(data[i + n_steps, target_col_index])
    return np.array(X), np.array(y)


unscaled_iv_values = []

target_col_names = ['IV_4500', 'IV_4550', 'IV_4600', 'IV_4650', 'IV_4700', 'IV_4750', 'IV_4800', 'IV_4850', 'IV_4900', 'IV_4925', 'IV_5000']
other_columns = ['Close', 'Delta', 'Theta', 'Gamma', 'Open Interest', 'Rho',
'Vega', 'Log Forward Moneyness']

# List to store metrics for each strike price
all_metrics = []

# Iterating through each CSV file
for target_col_name in target_col_names:
    # Reading each pivot DataFrame
    csv_file_name = f'pivot_df_{target_col_name[3:]}.csv'
    df = pd.read_csv(csv_file_name)
    
    df.drop(columns=['Exchange Date', 'Strike Price'], inplace=True)
    
    # Defining the features to scale
    features_to_scale = ['Days to Maturity','Close_Index', 'IV_4500',
           'IV_4550', 'IV_4600', 'IV_4650', 'IV_4700', 'IV_4750', 'IV_4800',
           'IV_4850', 'IV_4900', 'IV_4925', 'IV_5000', 'Close','Delta', 'Theta', 'Gamma', 'Open Interest','Rho', 'Vega', 'Log Forward Moneyness']

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features_to_scale])
    
    # Finding the target column index
    target_col_index = df[features_to_scale].columns.get_loc(target_col_name)
    #print(target_col_index)

    X, y = create_sequences(scaled_data, target_col_index, n_steps)
    un_scaled_X, unscaled_y = create_sequences(df.values, target_col_index, n_steps)
    
    # Splitting the data into training and validation sets
    split_index = int(len(X) * 0.5)
    #split_index = int(len(X) * 0.7)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]
    

    # LSTM Model
    model = Sequential([
        LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), verbose=1)
    
    # Predicting the last value in the validation set
    last_X_val = X_val[-1].reshape(1, n_steps, X_val.shape[2])
    last_predicted_iv = model.predict(last_X_val)[0][0]
    
    # Predict with LSTM Model on the Validation Set
    y_pred_lstm = model.predict(X_val)
    
    unscaled_actual = unscaled_y[split_index:]
    
    # Creating a dummy array with the same shape as the original scaled dataset
    dummy_array = np.zeros((len(y_pred_lstm), scaled_data.shape[1]))
    
    # Inserting the scaled predictions into the correct column
    dummy_array[:, target_col_index] = y_pred_lstm.flatten()
    
    # Inverse transforming the entire dummy array
    unscaled_predictions = scaler.inverse_transform(dummy_array)[:, target_col_index]

    # Creating an array of zeros with the shape of the original data
    last_predicted_iv_reshaped = np.zeros_like(scaled_data[-1])
    
    # Replacing the last element with last_predicted_iv
    last_predicted_iv_reshaped[target_col_index] = last_predicted_iv
    
    # Inverse transforming the predicted IV value to its original scale
    predicted_iv_unscaled = scaler.inverse_transform([last_predicted_iv_reshaped])
    
    # Extracting the unscaled predicted IV value
    unscaled_iv_value = predicted_iv_unscaled[0][target_col_index]
    unscaled_iv_values.append(unscaled_iv_value)
    
    print(f'Last predicted IV (unscaled) by LSTM for {target_col_name}: {unscaled_iv_value}')
    
    # Performance metrics
    mse_lstm = mean_squared_error(unscaled_actual, unscaled_predictions)

    rmse_lstm = np.sqrt(mse_lstm)
    
    mae_lstm = np.mean(np.abs(unscaled_actual - unscaled_predictions))
    
    mape_lstm = np.mean(np.abs((unscaled_actual - unscaled_predictions) / unscaled_actual)) * 100
    
    # Summarizing model performance
    performance_summary = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'MAPE'],
        'Value': [mse_lstm, rmse_lstm, mae_lstm, mape_lstm]
    })
    
    print(performance_summary)
    
    # Append metrics to the list
    all_metrics.append({
        'MSE': mse_lstm,
        'RMSE': rmse_lstm,
        'MAE': mae_lstm,
        'MAPE': mape_lstm
    })
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title(f'Model Loss for {target_col_name}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
    # Plotting actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(unscaled_actual, label='Actual IV', color='blue')
    plt.plot(unscaled_predictions, label='Predicted IV', color='red', alpha=0.7)
    plt.title('Actual vs. Predicted Implied Volatility')
    plt.xlabel('Time Steps')
    plt.ylabel('Implied Volatility')
    plt.legend()
    plt.show()



# Plotting IV values and target column names
plt.figure(figsize=(10, 6))
plt.plot(target_col_names, unscaled_iv_values)
plt.title('Predicted Implied Volatility for Next Trading Day')
plt.xlabel('Target Column Name')
plt.ylabel('Implied Volatility')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

'''
# Linear regression Model
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

model2 = LinearRegression()
model2.fit(X_train_flat, y_train)

y_pred2 = model2.predict(X_val_flat)
mse_autoregressive = mean_squared_error(y_val, y_pred2)


print(f'Last predicted IV by LSTM for {target_col_name}: {last_predicted_iv}')
print(f'MSE for {target_col_name} using LSTM: {mse_lstm}')
print(f'MSE for {target_col_name} using Linear Regression: {mse_autoregressive}')
'''
# Calculating the average of each metric through all strike prices
average_metrics = pd.DataFrame(all_metrics).mean().to_dict()

print("Average Performance Metrics Across All Strike Prices:")
print(average_metrics)

# Plotting the average metrics for visualization
plt.figure(figsize=(10, 6))
metrics_names = list(average_metrics.keys())
average_values = list(average_metrics.values())
plt.bar(metrics_names, average_values, color=['blue', 'orange', 'green', 'red'])
plt.title('Average Performance Metrics Across All Strike Prices')
plt.ylabel('Metric Values')
plt.show()

# Null model comparison - Autoregressive

from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import numpy as np

# Initializing a list to store MSE for each strike price
mse_list_ar = []
rmse_list_ar = []
mae_list_ar = []
mape_list_ar = []

for target_col_name in target_col_names:
    # Reading each pivot DataFrame
    csv_file_name = f'pivot_df_{target_col_name[3:]}.csv'
    df = pd.read_csv(csv_file_name)
    
    df.drop(columns=['Exchange Date', 'Strike Price'], inplace=True)
    
    features_to_scale = ['Days to Maturity','Close_Index', 'IV_4500',
           'IV_4550', 'IV_4600', 'IV_4650', 'IV_4700', 'IV_4750', 'IV_4800',
           'IV_4850', 'IV_4900', 'IV_4925', 'IV_5000', 'Close','Delta', 'Theta', 'Gamma', 'Open Interest','Rho', 'Vega', 'Log Forward Moneyness']

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features_to_scale])

    # Finding the target column index
    target_col_index = df[features_to_scale].columns.get_loc(target_col_name)
    print(target_col_index)

    X, y = create_sequences(scaled_data, target_col_index, n_steps)
    un_scaled_X, unscaled_y = create_sequences(df.values, target_col_index, n_steps)
    
    # Splitting the data into training and validation sets
    split_index = int(len(X) * 0.5)
    #split_index = int(len(X) * 0.7)
    X_train, X_val = un_scaled_X[:split_index], un_scaled_X[split_index:]
    y_train, y_val = unscaled_y[:split_index], unscaled_y[split_index:]

    # Combining training and validation sets for AR modeling
    y_combined = np.concatenate((y_train, y_val))
    
    # Fitting the Autoregressive model for the current strike price
    lag = 8
    model_ar = AutoReg(unscaled_y, lags=lag)
    model_ar_fitted = model_ar.fit()
    
    # Predicting the next values
    start = len(y_train)
    end = len(y_combined) - 1
    predictions_ar = model_ar_fitted.predict(start=start, end=end, dynamic=True)
    
    # Calculating metrics for the current strike price
    mse_ar = mean_squared_error(y_val, predictions_ar[:len(y_val)])
    rmse_ar = np.sqrt(mse_lstm)
    mae_ar = np.mean(np.abs(y_val - predictions_ar[:len(y_val)]))
    mape_ar = np.mean(np.abs((y_val - predictions_ar[:len(y_val)]) / y_val)) * 100
    
    mse_list_ar.append(mse_ar)
    rmse_list_ar.append(rmse_ar)
    mae_list_ar.append(mae_ar)
    mape_list_ar.append(mape_ar)
    
    print(f'MSE for {target_col_name} using AR: {mse_ar}')
    print(f'RMSE for {target_col_name} using AR: {rmse_ar}')
    print(f'MAE for {target_col_name} using AR: {mae_ar}')
    print(f'MAPE for {target_col_name} using AR: {mape_ar}')

# Calculating the average metrics across all strike prices
average_mse_ar = np.mean(mse_list_ar)
print(f'Average MSE across all strike prices using AR: {average_mse_ar}')
average_rmse_ar = np.mean(rmse_list_ar)
print(f'Average RMSE across all strike prices using AR: {average_rmse_ar}')
average_mae_ar = np.mean(mae_list_ar)
print(f'Average MAE across all strike prices using AR: {average_mae_ar}')
average_mape_ar = np.mean(mape_list_ar)
print(f'Average MAPE across all strike prices using AR: {average_mape_ar}')


max_lag = 25  # Maximum number of lags for testing
mse_values = []
lags = range(1, max_lag + 1)

for lag in lags:
    model_ar = AutoReg(y_combined, lags=lag)
    model_ar_fitted = model_ar.fit()
    
    # Predicting using the fitted model
    predictions_ar = model_ar_fitted.predict(start=len(y_train), end=len(y_combined)-1, dynamic=True)
    
    # Calculating MSE for this lag
    mse = mean_squared_error(y_val, predictions_ar[:len(y_val)])
    mse_values.append(mse)

# Plotting MSE against lags
plt.figure(figsize=(10, 6))
plt.plot(lags, mse_values, marker='o', linestyle='-', color='b')
plt.title('MSE vs. Lags for Autoregressive Model')
plt.xlabel('Lags')
plt.ylabel('MSE')
plt.xticks(lags)
plt.grid(True)
plt.show()
