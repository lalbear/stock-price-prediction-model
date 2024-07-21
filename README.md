## Stock Price Prediction with LSTM

This project aims to predict stock prices using a Long Short-Term Memory (LSTM) neural network. It leverages historical stock data to forecast future prices, employing techniques such as moving averages (20MA) and Relative Strength Index (RSI) to enhance the predictive model.

### Features

1. **Data Fetching**: Utilizes the Alpha Vantage API to fetch historical daily adjusted stock prices. The fetched data is processed to include only the necessary features such as closing prices.
2. **Feature Engineering**: Adds technical indicators like the 20-day moving average (20MA) and Relative Strength Index (RSI) to the dataset. These features provide insights into stock trends and momentum.
3. **Data Preparation**: Transforms the time series data into windowed datasets suitable for training the LSTM model. This involves creating sequences of past stock prices to predict future values.
4. **Model Training**: Trains an LSTM model using TensorFlow/Keras. The model architecture includes LSTM layers followed by dense layers, optimized for predicting stock prices.
5. **Evaluation**: Splits the data into training, validation, and test sets to evaluate the model's performance. It also predicts future stock prices and visualizes the results.
6. **Future Prediction**: Generates future dates and predicts stock prices for those dates based on the trained model.

### Requirements

- Python 3.x
- TensorFlow
- Pandas
- Numpy
- Matplotlib
- Requests
This project provides a foundation for stock price prediction, showcasing the application of LSTM networks in financial time series analysis.
