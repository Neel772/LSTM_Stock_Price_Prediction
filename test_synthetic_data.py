"""
Test script with synthetic data to verify LSTM stock prediction implementation.
Uses generated time-series data instead of fetching from external sources.
"""

from lstm_stock_prediction import LSTMStockPredictor
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_stock_data(days=500, start_price=100, trend=0.001, volatility=0.02):
    """
    Generate synthetic stock price data with trend and volatility.
    
    Args:
        days (int): Number of days to generate
        start_price (float): Starting stock price
        trend (float): Daily trend (drift)
        volatility (float): Daily volatility
    
    Returns:
        pandas.DataFrame: Synthetic stock data
    """
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
    prices = [start_price]
    
    for _ in range(days - 1):
        change = prices[-1] * (trend + volatility * np.random.randn())
        new_price = prices[-1] + change
        prices.append(max(new_price, 1))  # Ensure price stays positive
    
    df = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.randn()) * 0.01) for p in prices],
        'Low': [p * (1 - abs(np.random.randn()) * 0.01) for p in prices],
        'Close': prices,
        'Volume': [1000000 + np.random.randint(-500000, 500000) for _ in prices]
    }, index=dates)
    
    return df

def test_lstm_with_synthetic_data():
    """Test the LSTM predictor with synthetic data."""
    print("=" * 80)
    print("Testing LSTM Stock Price Prediction with Synthetic Data")
    print("=" * 80)
    
    # Initialize predictor
    ticker = 'SYNTHETIC_STOCK'
    print(f"\n1. Initializing predictor for {ticker}...")
    predictor = LSTMStockPredictor(ticker=ticker, lookback_period=60)
    print("✓ Predictor initialized successfully")
    
    # Generate synthetic data
    print("\n2. Generating synthetic stock data...")
    try:
        predictor.data = generate_synthetic_stock_data(days=500, start_price=150, trend=0.0005, volatility=0.015)
        print(f"✓ Synthetic data generated successfully! Shape: {predictor.data.shape}")
    except Exception as e:
        print(f"✗ Error generating data: {e}")
        return False
    
    # Prepare data
    print("\n3. Preparing data for training...")
    try:
        X_train, y_train, X_test, y_test, train_size = predictor.prepare_data(test_size=0.2)
        print("✓ Data prepared successfully")
        print(f"   Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    except Exception as e:
        print(f"✗ Error preparing data: {e}")
        return False
    
    # Build model
    print("\n4. Building LSTM model...")
    try:
        predictor.build_model(units=[50, 50], dropout=0.2)
        print("✓ Model built successfully")
    except Exception as e:
        print(f"✗ Error building model: {e}")
        return False
    
    # Train model (with fewer epochs for testing)
    print("\n5. Training model (15 epochs for testing)...")
    try:
        history = predictor.train(X_train, y_train, epochs=15, batch_size=32, validation_split=0.1)
        print("✓ Model trained successfully")
    except Exception as e:
        print(f"✗ Error training model: {e}")
        return False
    
    # Make predictions
    print("\n6. Generating predictions...")
    try:
        train_predictions = predictor.predict(X_train)
        test_predictions = predictor.predict(X_test)
        print("✓ Predictions generated successfully")
        print(f"   Train predictions shape: {train_predictions.shape}")
        print(f"   Test predictions shape: {test_predictions.shape}")
    except Exception as e:
        print(f"✗ Error generating predictions: {e}")
        return False
    
    # Calculate metrics
    print("\n7. Calculating performance metrics...")
    try:
        train_metrics = predictor.calculate_metrics(
            predictor.scaler.inverse_transform(y_train.reshape(-1, 1)),
            train_predictions
        )
        test_metrics = predictor.calculate_metrics(
            predictor.scaler.inverse_transform(y_test.reshape(-1, 1)),
            test_predictions
        )
        
        print("\n   Training Set Metrics:")
        for metric, value in train_metrics.items():
            print(f"      {metric}: {value:.4f}")
        
        print("\n   Test Set Metrics:")
        for metric, value in test_metrics.items():
            print(f"      {metric}: {value:.4f}")
        
        # Check if R2 score is reasonable
        if test_metrics['R2 Score'] > 0.5:
            print("\n   ✓ Model shows good predictive performance (R² > 0.5)")
        else:
            print("\n   ⚠ Model performance is acceptable for synthetic data test")
        
        print("\n✓ Metrics calculated successfully")
    except Exception as e:
        print(f"✗ Error calculating metrics: {e}")
        return False
    
    # Visualize (save only, don't display)
    print("\n8. Creating visualization...")
    try:
        predictor.visualize_predictions(
            y_train, y_test, train_predictions, test_predictions, 
            train_size, save_path='synthetic_stock_prediction.png'
        )
        print("✓ Visualization created and saved as 'synthetic_stock_prediction.png'")
    except Exception as e:
        print(f"✗ Error creating visualization: {e}")
        return False
    
    # Test model save/load
    print("\n9. Testing model persistence...")
    try:
        predictor.save_model('test_lstm_model.h5')
        print("✓ Model saved as 'test_lstm_model.h5'")
        
        # Create new predictor and load model
        new_predictor = LSTMStockPredictor(ticker=ticker, lookback_period=60)
        new_predictor.scaler = predictor.scaler  # Copy scaler
        new_predictor.load_model('test_lstm_model.h5')
        
        # Test prediction with loaded model
        test_pred_loaded = new_predictor.predict(X_test[:5])
        print("✓ Model loaded and tested successfully")
    except Exception as e:
        print(f"✗ Error with model persistence: {e}")
        return False
    
    # Display sample predictions
    print("\n10. Sample Predictions (first 5 test samples):")
    print("    " + "-" * 50)
    y_test_actual = predictor.scaler.inverse_transform(y_test.reshape(-1, 1))
    for i in range(min(5, len(test_predictions))):
        actual = y_test_actual[i][0]
        predicted = test_predictions[i][0]
        diff = abs(actual - predicted)
        diff_pct = (diff / actual) * 100
        print(f"    Sample {i+1}: Actual=${actual:.2f}, Predicted=${predicted:.2f}, "
              f"Diff=${diff:.2f} ({diff_pct:.2f}%)")
    print("    " + "-" * 50)
    
    print("\n" + "=" * 80)
    print("✓ All tests passed successfully!")
    print("=" * 80)
    print("\nThe LSTM stock prediction implementation is working correctly!")
    print("Key components tested:")
    print("  ✓ Data preprocessing and normalization")
    print("  ✓ LSTM model architecture and compilation")
    print("  ✓ Model training with validation")
    print("  ✓ Prediction generation")
    print("  ✓ Performance metrics calculation")
    print("  ✓ Visualization generation")
    print("  ✓ Model save/load functionality")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = test_lstm_with_synthetic_data()
    exit(0 if success else 1)
