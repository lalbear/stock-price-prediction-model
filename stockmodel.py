import sys
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import os
from sklearn.preprocessing import MinMaxScaler

# Set console encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def fetch_recent_stock_data(symbol, api_key, output_size='full'):
    url = f'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'outputsize': output_size,
        'apikey': api_key,
    }
    response = requests.get(url, params=params)
    data = response.json()

    if 'Time Series (Daily)' not in data:
        print("API response:", data)
        raise ValueError("Error fetching data from Alpha Vantage API: " + data.get("Note", "No detailed error message available."))

    time_series = data['Time Series (Daily)']
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df = df.rename(columns={'4. close': 'Adj Close'}).astype(float)
    
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    return df[['Adj Close']]  # Return only the 'Adj Close' column

def add_features(df):
    df['20MA'] = df['Adj Close'].rolling(window=20).mean()
    df['50MA'] = df['Adj Close'].rolling(window=50).mean()

    delta = df['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df.dropna()

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

try:
    # Load historical data from Excel
    historical_df = pd.read_excel('AAPL_historical.xlsx', index_col='Date')
    historical_df = historical_df[['Adj Close']]  # Ensure we only have the 'Adj Close' column
    historical_df = add_features(historical_df)

    print("Historical data:")
    print(historical_df.head())
    print(historical_df.tail())

    # Prepare data for LSTM
    seq_length = 60  # Use 60 days of historical data to predict the next day
    features = ['Adj Close', '20MA', '50MA', 'RSI']
    
    data = historical_df[features]
    X, y = create_sequences(data.values, seq_length)
    y = y[:, 0]  # We only want to predict the 'Adj Close' price

    # Normalize the data
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_scaled = feature_scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    # Build the model
    model = Sequential([
        layers.LSTM(64, input_shape=(seq_length, len(features))),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Train the model
    history = model.fit(X_scaled, y_scaled, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

    # Fetch recent data for predictions
    api_key = 'NQYWXGC8LMPJG1CZ'  # Your actual API key
    symbol = 'AAPL'               # Stock symbol for Apple Inc.
    recent_df = fetch_recent_stock_data(symbol, api_key, output_size='full')
    recent_df = add_features(recent_df)

    print("\nRecent data:")
    print(recent_df.tail())

    # Prepare recent data for prediction
    recent_data = recent_df[features].values
    recent_scaled = feature_scaler.transform(recent_data)

    # Combine historical and recent data if necessary
    combined_data = np.vstack((X_scaled[-1], recent_scaled))
    if len(combined_data) < seq_length:
        padding = X_scaled[-(seq_length - len(combined_data)):]
        combined_data = np.vstack((padding, combined_data))

    # Ensure we have the correct sequence length
    last_sequence = combined_data[-seq_length:]

    # Make predictions
    future_predictions = []

    for _ in range(5):  # Predict next 5 days
        next_pred = model.predict(last_sequence.reshape(1, seq_length, len(features))).flatten()[0]
        future_predictions.append(next_pred)
        last_sequence = np.roll(last_sequence, -1, axis=0)
        last_sequence[-1] = [next_pred] + list(last_sequence[-1, 1:])  # Keep other features unchanged

    # Inverse transform the predictions
    future_predictions = target_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(recent_df.index, recent_df['Adj Close'], label='Actual')
    
    future_dates = pd.date_range(start=recent_df.index[-1] + pd.Timedelta(days=1), periods=5)
    plt.plot(future_dates, future_predictions, label='Future Predictions')
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('AAPL Stock Price Prediction')
    plt.legend()
    plt.show()

    print("\nPredicted Prices for Future Dates:")
    for date, price in zip(future_dates, future_predictions):
        print(f"{date.date()}: {price:.2f}")

except Exception as e:
    print("An error occurred:", str(e))
