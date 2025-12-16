"""
LSTM Stock Price Prediction
A deep learning model using LSTM to forecast stock prices based on historical data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import os


class LSTMStockPredictor:
    """
    LSTM-based stock price predictor for time-series forecasting.
    """
    
    def __init__(self, ticker='AAPL', lookback_period=60):
        """
        Initialize the LSTM Stock Predictor.
        
        Args:
            ticker (str): Stock ticker symbol (default: 'AAPL')
            lookback_period (int): Number of previous days to use for prediction (default: 60)
        """
        self.ticker = ticker
        self.lookback_period = lookback_period
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.train_data = None
        self.test_data = None
        
    def fetch_data(self, start_date=None, end_date=None):
        """
        Fetch historical stock data from Yahoo Finance.
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"Fetching data for {self.ticker} from {start_date} to {end_date}...")
        self.data = yf.download(self.ticker, start=start_date, end=end_date)
        
        if self.data.empty:
            raise ValueError(f"No data found for ticker {self.ticker}")
            
        print(f"Data fetched successfully! Shape: {self.data.shape}")
        return self.data
    
    def prepare_data(self, test_size=0.2):
        """
        Prepare and preprocess data for training.
        
        Args:
            test_size (float): Proportion of data to use for testing (default: 0.2)
        """
        if self.data is None:
            raise ValueError("No data available. Please fetch data first using fetch_data()")
        
        # Use 'Close' price for prediction
        close_prices = self.data['Close'].values.reshape(-1, 1)
        
        # Normalize the data
        scaled_data = self.scaler.fit_transform(close_prices)
        
        # Calculate split point
        train_size = int(len(scaled_data) * (1 - test_size))
        
        # Create training sequences
        X_train, y_train = self._create_sequences(scaled_data[:train_size])
        X_test, y_test = self._create_sequences(scaled_data[train_size - self.lookback_period:])
        
        print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Testing data shape: X={X_test.shape}, y={y_test.shape}")
        
        return X_train, y_train, X_test, y_test, train_size
    
    def _create_sequences(self, data):
        """
        Create sequences for LSTM training.
        
        Args:
            data (np.array): Normalized price data
            
        Returns:
            X, y: Input sequences and target values
        """
        X, y = [], []
        for i in range(self.lookback_period, len(data)):
            X.append(data[i - self.lookback_period:i, 0])
            y.append(data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def build_model(self, units=[50, 50], dropout=0.2):
        """
        Build the LSTM model architecture.
        
        Args:
            units (list): Number of LSTM units in each layer
            dropout (float): Dropout rate for regularization
        """
        self.model = Sequential()
        
        # First LSTM layer
        self.model.add(LSTM(units=units[0], return_sequences=True, 
                           input_shape=(self.lookback_period, 1)))
        self.model.add(Dropout(dropout))
        
        # Additional LSTM layers
        for i in range(1, len(units)):
            return_seq = i < len(units) - 1
            self.model.add(LSTM(units=units[i], return_sequences=return_seq))
            self.model.add(Dropout(dropout))
        
        # Output layer
        self.model.add(Dense(units=1))
        
        # Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        print("Model built successfully!")
        self.model.summary()
        
        return self.model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training input sequences
            y_train: Training target values
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Proportion of training data for validation
        """
        if self.model is None:
            raise ValueError("Model not built. Please call build_model() first.")
        
        print(f"Training model for {epochs} epochs...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        print("Training completed!")
        return history
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Input sequences
            
        Returns:
            Predictions in original scale
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        predictions = self.model.predict(X)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions
    
    def visualize_predictions(self, y_train, y_test, train_predictions, test_predictions, 
                            train_size, save_path='stock_prediction.png'):
        """
        Visualize the actual vs predicted stock prices.
        
        Args:
            y_train: Actual training values
            y_test: Actual test values
            train_predictions: Predicted training values
            test_predictions: Predicted test values
            train_size: Index where test data starts
            save_path (str): Path to save the visualization
        """
        # Inverse transform the actual values
        y_train_actual = self.scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Create plot
        plt.figure(figsize=(16, 8))
        
        # Plot training data
        train_dates = self.data.index[self.lookback_period:train_size]
        plt.plot(train_dates, y_train_actual, label='Actual Training Data', color='blue', alpha=0.6)
        plt.plot(train_dates, train_predictions, label='Predicted Training Data', color='green', alpha=0.6)
        
        # Plot test data
        test_dates = self.data.index[train_size + self.lookback_period:]
        plt.plot(test_dates, y_test_actual, label='Actual Test Data', color='orange', alpha=0.8)
        plt.plot(test_dates, test_predictions, label='Predicted Test Data', color='red', alpha=0.8)
        
        plt.title(f'{self.ticker} Stock Price Prediction using LSTM', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Stock Price (USD)', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        plt.show()
        
    def calculate_metrics(self, y_actual, y_predicted):
        """
        Calculate performance metrics.
        
        Args:
            y_actual: Actual values
            y_predicted: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_actual, y_predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_actual, y_predicted)
        r2 = r2_score(y_actual, y_predicted)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2 Score': r2
        }
        
        return metrics
    
    def save_model(self, filepath='lstm_stock_model.h5'):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='lstm_stock_model.h5'):
        """
        Load a saved model.
        
        Args:
            filepath (str): Path to the saved model
        """
        from tensorflow.keras.models import load_model
        
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")


def main():
    """
    Main function to demonstrate the LSTM stock prediction pipeline.
    """
    print("=" * 80)
    print("LSTM Stock Price Prediction System")
    print("=" * 80)
    
    # Initialize predictor
    ticker = 'AAPL'  # Apple Inc.
    predictor = LSTMStockPredictor(ticker=ticker, lookback_period=60)
    
    # Fetch data (last 5 years)
    predictor.fetch_data()
    
    # Prepare data
    X_train, y_train, X_test, y_test, train_size = predictor.prepare_data(test_size=0.2)
    
    # Build model
    predictor.build_model(units=[50, 50], dropout=0.2)
    
    # Train model
    history = predictor.train(X_train, y_train, epochs=50, batch_size=32)
    
    # Make predictions
    print("\nGenerating predictions...")
    train_predictions = predictor.predict(X_train)
    test_predictions = predictor.predict(X_test)
    
    # Calculate metrics
    print("\n" + "=" * 80)
    print("Model Performance Metrics")
    print("=" * 80)
    
    train_metrics = predictor.calculate_metrics(
        predictor.scaler.inverse_transform(y_train.reshape(-1, 1)),
        train_predictions
    )
    test_metrics = predictor.calculate_metrics(
        predictor.scaler.inverse_transform(y_test.reshape(-1, 1)),
        test_predictions
    )
    
    print("\nTraining Set Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTest Set Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Visualize results
    print("\n" + "=" * 80)
    print("Generating Visualization...")
    print("=" * 80)
    predictor.visualize_predictions(
        y_train, y_test, train_predictions, test_predictions, train_size
    )
    
    # Save model
    predictor.save_model('lstm_stock_model.h5')
    
    print("\n" + "=" * 80)
    print("Process completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
