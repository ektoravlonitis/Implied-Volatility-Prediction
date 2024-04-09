# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 21:17:08 2024

@author: he_98
"""

import pandas as pd
from datetime import datetime
from numpy import log, exp

# Defining the fixed maturity date
maturity_date = datetime(2024, 4, 19)

# Defining the list of CSV filenames
csv_filenames = ['4500.csv', '4550.csv', '4600.csv', '4650.csv', '4700.csv', '4750.csv', '4800.csv', '4850.csv', '4900.csv', '4925.csv', '5000.csv']  # Replace these with your actual CSV file names

# Function to process a single CSV file
def process_csv(file_path):
    df = pd.read_csv(file_path)
    strike_price = int(file_path.replace('.csv', ''))

    df.replace({'\+': '', '%': ''}, regex=True, inplace=True)
    if df['Open Interest'].dtype == object:
        #print(file_path)
        df['Open Interest'] = df['Open Interest'].str.replace(',', '').astype(float)
    df['Exchange Date'] = pd.to_datetime(df['Exchange Date'], format='%d-%b-%Y')
    '''
    for col in df.columns:
        if col not in ['Exchange Date', 'Implied Volatility', 'Strike Price']:
            df.drop(col, axis=1, inplace=True)
    '''
    for col in ['Volume', '%CVol', 'O-C', 'H-L', 'Net', '%Chg', 'Ask Implied Volatility', 'Bid Implied Volatility', 'Open', 'High', 'Low', 'Bid', 'Ask', 'Mid Price']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    df.sort_values(by='Exchange Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df['Strike Price'] = strike_price
    
    # Calculating 'Days to Maturity' and adding the column
    df['Days to Maturity'] = (maturity_date - df['Exchange Date']).dt.days
    
    return df

# Dictionary to hold the processed DataFrames
processed_dfs = {}

# Looping through the CSV filenames
for csv_filename in csv_filenames:
    # Processing each CSV file and storing the result in the dictionary
    processed_dfs[csv_filename] = process_csv(csv_filename)


# Concatenating all DataFrames in the dictionary into a single DataFrame
combined_df = pd.concat(processed_dfs.values(), ignore_index=True)

# Sorting the combined DataFrame by 'Exchange Date'
combined_df.sort_values(by='Exchange Date', inplace=True)

# Resetting index after sorting
combined_df.reset_index(drop=True, inplace=True)


# Adding the Euro STOXX 50 Close Price
index = pd.read_csv('eurostoxx.csv')
index['Close_Index'] = index['Close'].str.replace(',', '').astype(float)

index['Exchange Date'] = pd.to_datetime(index['Exchange Date'], format='%d-%b-%Y')

# Keeping only 'Exchange Date' and 'Close' columns
index = index[['Exchange Date', 'Close_Index']]

# Merging the index 'Close' price into combined_df based on 'Exchange Date'
combined_df = pd.merge(combined_df, index, on='Exchange Date', how='left')

# Removing rows where either 'Implied Volatility' or 'Close' column has NaN values
combined_df = combined_df.dropna(subset=['Implied Volatility'])

# Calculating the percentage of NaN values in each column
nan_percentage = combined_df.isnull().mean() * 100

# Printing the percentage of NaN values for each column
print(nan_percentage)
# Converting the percentages to a DataFrame for better formatting
nan_percentage_df = nan_percentage.reset_index()
nan_percentage_df.columns = ['Column Name', 'Percentage of NaN Values']

# Saving the table to an excel file
nan_percentage_df.to_excel('nan_percentages.xlsx', index=False, float_format='%.2f')
print(nan_percentage_df)

# Uncomment the rows in the for loop to get the main_columns.csv, then comment them and rerun it again for the rest
# combined_df.to_csv('main_columns.csv', index=False)


# The risk-free rate obtained from the Euro area’s 1-year maturity Triple-A rated government bond yield curve
r = 0.0327

# Directly calculating 'Log Forward Moneyness'
combined_df['Log Forward Moneyness'] = log(combined_df['Strike Price'] / (combined_df['Close_Index'] * exp(r * (combined_df['Days to Maturity'] / 365))))
#combined_df['Moneyness'] = combined_df['Strike Price'] / combined_df['Close_Index']

# Saveing the modified DataFrame back to a new CSV file
combined_df.to_csv('combined_data.csv', index=False)

# Keeping dates with data across all strike prices
def filter_max_dates_csv(file_path):
    df = pd.read_csv(file_path)
    date_counts = df.iloc[:, 0].value_counts()
    
    # Finding the maximum count
    max_count = date_counts.max()
    
    # Filtering dates with the maximum count
    max_dates = date_counts[date_counts == max_count].index.tolist()
    
    # Filtering the DataFrame for those dates
    filtered_df = df[df.iloc[:, 0].isin(max_dates)]
    
    return filtered_df

file_path = 'main_columns.csv'
filtered_data = filter_max_dates_csv(file_path)
filtered_data.to_csv('main_columns_filtered.csv', index=False)