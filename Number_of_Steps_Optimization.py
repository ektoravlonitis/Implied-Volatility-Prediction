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

# Load the combined_data.csv data
comb_df = pd.read_csv('combined_data.csv')

# Get unique strike prices from combined_data.csv
unique_strike_prices = comb_df['Strike Price'].unique()

# For each unique strike price, we filter comb_df and update pivot_df accordingly
for strike_price in unique_strike_prices:
    filtered_comb_df = comb_df[comb_df['Strike Price'] == strike_price]
    merged_df = pivot_df.merge(filtered_comb_df, how='left', on='Exchange Date')
    merged_df = merged_df.drop(columns=['Implied Volatility'])

    for col in ['Days to Maturity', 'Close_Index']:
        if f'{col}_x' in merged_df.columns and f'{col}_y' in merged_df.columns:
            merged_df = merged_df.drop(columns=[f'{col}_y'])
            merged_df = merged_df.rename(columns={f'{col}_x': col})

    merged_df.to_csv(f'pivot_df_{strike_price}.csv', index=False)


# Function to create sequences for LSTM
def create_sequences(data, target_col_index, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i, :])
        y.append(data[i, target_col_index])
    return np.array(X), np.array(y)

# List of different numbers of steps to try
n_steps_list = [1, 5, 10, 20, 30]

# Recording the performance for each number of steps
performance_records = {}

for n_steps in n_steps_list:
    unscaled_iv_values = []

    target_col_names = ['IV_4500', 'IV_4550', 'IV_4600', 'IV_4650', 'IV_4700', 'IV_4750', 'IV_4800', 'IV_4850', 'IV_4900', 'IV_4925', 'IV_5000']
    other_columns = ['Close', 'Delta', 'Theta', 'Gamma', 'Open Interest', 'Rho',
    'Vega', 'Log Forward Moneyness']
    
    # Iterating through each CSV file
    for target_col_name in target_col_names:
        csv_file_name = f'pivot_df_{target_col_name[3:]}.csv'
        df = pd.read_csv(csv_file_name)

        df.drop(columns=['Exchange Date', 'Strike Price'], inplace=True)
        
        features_to_scale = ['Days to Maturity','Close_Index', 'IV_4500',
               'IV_4550', 'IV_4600', 'IV_4650', 'IV_4700', 'IV_4750', 'IV_4800',
               'IV_4850', 'IV_4900', 'IV_4925', 'IV_5000', 'Close','Delta', 'Theta', 'Gamma', 'Open Interest','Rho', 'Vega', 'Log Forward Moneyness']
    
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[features_to_scale])
          
        target_col_index = df[features_to_scale].columns.get_loc(target_col_name)
        print(target_col_index)
        
        X, y = create_sequences(scaled_data, target_col_index, n_steps)
        
        # Splitting the data into training and validation sets
        split_index = int(len(X) * 0.5)
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
        
        # Predicting with LSTM Model on the Validation Set
        y_pred_lstm = model.predict(X_val)
        
        # Calculating the metrics for the LSTM model
        mse_lstm = mean_squared_error(y_val, y_pred_lstm)
                
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
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.title(f'Model Loss for {target_col_name}')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()
    
    
    # Plotting IV values and target column names
    plt.figure(figsize=(10, 6))
    plt.plot(target_col_names, unscaled_iv_values)
    plt.title('Predicted Implied Volatility for Next Trading Day')
    plt.xlabel('Target Column Name')
    plt.ylabel('Implied Volatility')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
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
    '''
    performance_records[n_steps] = mse_lstm
    
    print(f'Last predicted IV by LSTM for {target_col_name}: {last_predicted_iv}')
    print(f'MSE for {target_col_name} using LSTM: {mse_lstm}')
    #print(f'MSE for {target_col_name} using Linear Regression: {mse_autoregressive}')

# Printing the performance records
for n_steps, mse in performance_records.items():
    print(f'MSE for {n_steps} steps: {mse}')

# Identifying the best number of steps based on MSE
best_n_steps = min(performance_records, key=performance_records.get)
print(f'Best number of steps: {best_n_steps} with MSE: {performance_records[best_n_steps]}')

# Plotting the performance:
n_steps_values = list(performance_records.keys())
mse_values = list(performance_records.values())

# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(n_steps_values, mse_values, marker='o')
plt.title('LSTM Model Performance by Number of Steps')
plt.xlabel('Number of Steps (n_steps)')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True)
plt.xticks(n_steps_values)
plt.show()
