"""
Quick test script to verify LSTM stock prediction implementation.
Tests with minimal epochs for faster validation.
"""

from lstm_stock_prediction import LSTMStockPredictor
import numpy as np

def test_lstm_predictor():
    """Test the LSTM predictor with minimal configuration."""
    print("=" * 80)
    print("Testing LSTM Stock Price Prediction Implementation")
    print("=" * 80)
    
    # Initialize predictor
    ticker = 'AAPL'
    print(f"\n1. Initializing predictor for {ticker}...")
    predictor = LSTMStockPredictor(ticker=ticker, lookback_period=60)
    print("✓ Predictor initialized successfully")
    
    # Fetch data (last 2 years for faster testing)
    print("\n2. Fetching stock data...")
    try:
        predictor.fetch_data(start_date='2022-01-01', end_date='2024-01-01')
        print("✓ Data fetched successfully")
    except Exception as e:
        print(f"✗ Error fetching data: {e}")
        return False
    
    # Prepare data
    print("\n3. Preparing data...")
    try:
        X_train, y_train, X_test, y_test, train_size = predictor.prepare_data(test_size=0.2)
        print("✓ Data prepared successfully")
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
    print("\n5. Training model (10 epochs for testing)...")
    try:
        history = predictor.train(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
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
        
        print("\nTraining Set Metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nTest Set Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("✓ Metrics calculated successfully")
    except Exception as e:
        print(f"✗ Error calculating metrics: {e}")
        return False
    
    # Visualize (save only, don't display)
    print("\n8. Creating visualization...")
    try:
        predictor.visualize_predictions(
            y_train, y_test, train_predictions, test_predictions, 
            train_size, save_path='test_stock_prediction.png'
        )
        print("✓ Visualization created successfully")
    except Exception as e:
        print(f"✗ Error creating visualization: {e}")
        return False
    
    # Test model save/load
    print("\n9. Testing model persistence...")
    try:
        predictor.save_model('test_model.h5')
        print("✓ Model saved successfully")
        
        # Create new predictor and load model
        new_predictor = LSTMStockPredictor(ticker=ticker, lookback_period=60)
        new_predictor.scaler = predictor.scaler  # Copy scaler
        new_predictor.load_model('test_model.h5')
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error with model persistence: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_lstm_predictor()
    exit(0 if success else 1)
