# LSTM Stock Price Prediction

A deep learning project using Long Short-Term Memory (LSTM) neural networks to forecast stock prices based on historical time-series data. This project demonstrates the application of LSTM models for financial time-series prediction with comprehensive data visualization.

## 🎯 Features

- **Time-Series Forecasting**: Uses LSTM neural networks to predict future stock prices
- **Data Fetching**: Automatically downloads historical stock data using Yahoo Finance API
- **Data Preprocessing**: Normalizes data using MinMaxScaler for optimal training
- **Customizable Model**: Configurable LSTM architecture with multiple layers and dropout
- **Performance Metrics**: Calculates MSE, RMSE, MAE, and R² scores
- **Visualization**: Generates comprehensive plots comparing actual vs predicted prices
- **Model Persistence**: Save and load trained models for future predictions

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Neel772/LSTM_Stock_Price_Prediction.git
cd LSTM_Stock_Price_Prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Usage

### Basic Usage

Run the main script to train the model and generate predictions:

```bash
python lstm_stock_prediction.py
```

This will:
1. Fetch historical stock data (default: AAPL - Apple Inc.)
2. Preprocess and normalize the data
3. Build and train the LSTM model
4. Generate predictions
5. Display performance metrics
6. Create a visualization plot
7. Save the trained model

### Custom Usage

You can customize the prediction by modifying the parameters in the script or using the class directly:

```python
from lstm_stock_prediction import LSTMStockPredictor

# Initialize predictor with custom settings
predictor = LSTMStockPredictor(ticker='GOOGL', lookback_period=90)

# Fetch data for a specific date range
predictor.fetch_data(start_date='2020-01-01', end_date='2024-12-31')

# Prepare data with custom test split
X_train, y_train, X_test, y_test, train_size = predictor.prepare_data(test_size=0.3)

# Build model with custom architecture
predictor.build_model(units=[100, 50, 25], dropout=0.3)

# Train with custom parameters
history = predictor.train(X_train, y_train, epochs=100, batch_size=64)

# Make predictions
test_predictions = predictor.predict(X_test)

# Visualize results
predictor.visualize_predictions(y_train, y_test, 
                                predictor.predict(X_train), 
                                test_predictions, 
                                train_size)
```

## 🏗️ Model Architecture

The LSTM model consists of:
- **Input Layer**: Accepts sequences of historical prices (default: 60 days lookback)
- **LSTM Layers**: Multiple stacked LSTM layers (default: 2 layers with 50 units each)
- **Dropout Layers**: Regularization to prevent overfitting (default: 20% dropout)
- **Output Layer**: Dense layer with single unit for price prediction
- **Optimizer**: Adam optimizer
- **Loss Function**: Mean Squared Error (MSE)

## 📈 Output

The model generates:

1. **Console Output**: Training progress, performance metrics (RMSE, MAE, R²)
2. **Visualization Plot**: Graph showing actual vs predicted prices for both training and test data
3. **Saved Model**: Trained model saved as `lstm_stock_model.h5` for future use

## 🔧 Configuration

Key parameters you can adjust:

- `ticker`: Stock symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')
- `lookback_period`: Number of previous days used for prediction (default: 60)
- `test_size`: Proportion of data used for testing (default: 0.2)
- `units`: List of LSTM units per layer (default: [50, 50])
- `dropout`: Dropout rate (default: 0.2)
- `epochs`: Number of training epochs (default: 50)
- `batch_size`: Training batch size (default: 32)

## 📋 Requirements

- tensorflow>=2.13.0
- numpy>=1.24.3
- pandas>=2.0.3
- matplotlib>=3.7.2
- scikit-learn>=1.3.0
- yfinance>=0.2.28

## 🎓 How It Works

1. **Data Collection**: Historical stock prices are fetched from Yahoo Finance
2. **Preprocessing**: Data is normalized to [0,1] range using MinMaxScaler
3. **Sequence Creation**: Time-series data is converted into supervised learning format
4. **Model Training**: LSTM learns patterns from historical sequences
5. **Prediction**: Model forecasts future prices based on learned patterns
6. **Evaluation**: Performance is measured using multiple metrics
7. **Visualization**: Results are plotted for easy interpretation

## 📝 Example Output

```
Training Set Metrics:
  MSE: 12.3456
  RMSE: 3.5137
  MAE: 2.7891
  R2 Score: 0.9823

Test Set Metrics:
  MSE: 15.6789
  RMSE: 3.9598
  MAE: 3.1245
  R2 Score: 0.9756
```

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📄 License

This project is open source and available under the MIT License.

## ⚠️ Disclaimer

This project is for educational purposes only. Stock price prediction is inherently uncertain, and this model should not be used as the sole basis for investment decisions. Always consult with financial advisors and conduct thorough research before making investment decisions.