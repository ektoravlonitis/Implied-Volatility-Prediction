# Implied-Volatility-Prediction

## Introduction

This coursework explores the application of machine learning techniques to predict the implied volatility skew in financial markets. Focusing on Long Short-Term Memory (LSTM) networks, a model is developed that outperforms traditional volatility forecasting methods.
By analyzing various options within the Euro STOXX50index, the study demonstrates how machine learning can provide more accurate predictions of market dynamics, offering significant improvements over the Black-Scholes model.

## How to run

Run the python files in the following order:
1. Data Engineering: Start with Data_engineering.py to prepare your CSV data correctly.
2. Parameter Optimization: Use Grid_Search_Optimization.py and Number_of_Steps_Optimization.py to find the best model parameters and optimal time steps.
3. Prediction: Execute LSTM_model.py to build and run the LSTM prediction model with the optimized settings.
