"""
Example usage script for LSTM Stock Price Prediction.
This demonstrates how to use the LSTMStockPredictor class with different configurations.
"""

from lstm_stock_prediction import LSTMStockPredictor

def example_basic_usage():
    """
    Basic example: Train and predict with default settings.
    """
    print("="*80)
    print("Example 1: Basic Usage with Default Settings")
    print("="*80)
    
    # Initialize predictor
    predictor = LSTMStockPredictor(ticker='AAPL', lookback_period=60)
    
    # Fetch data
    predictor.fetch_data()
    
    # Prepare data
    X_train, y_train, X_test, y_test, train_size = predictor.prepare_data(test_size=0.2)
    
    # Build and train model
    predictor.build_model(units=[50, 50], dropout=0.2)
    predictor.train(X_train, y_train, epochs=50, batch_size=32)
    
    # Make predictions
    train_predictions = predictor.predict(X_train)
    test_predictions = predictor.predict(X_test)
    
    # Calculate and display metrics
    test_metrics = predictor.calculate_metrics(
        predictor.scaler.inverse_transform(y_test.reshape(-1, 1)),
        test_predictions
    )
    print("\nTest Set Performance:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Visualize
    predictor.visualize_predictions(
        y_train, y_test, train_predictions, test_predictions, train_size
    )
    
    # Save model
    predictor.save_model('apple_stock_model.h5')
    print("\nModel saved successfully!")


def example_custom_configuration():
    """
    Advanced example: Custom model configuration and parameters.
    """
    print("\n" + "="*80)
    print("Example 2: Custom Configuration")
    print("="*80)
    
    # Initialize with custom settings
    predictor = LSTMStockPredictor(ticker='GOOGL', lookback_period=90)
    
    # Fetch specific date range
    predictor.fetch_data(start_date='2019-01-01', end_date='2023-12-31')
    
    # Custom train-test split
    X_train, y_train, X_test, y_test, train_size = predictor.prepare_data(test_size=0.25)
    
    # Build deeper model with more units
    predictor.build_model(units=[100, 75, 50], dropout=0.3)
    
    # Train with more epochs
    predictor.train(X_train, y_train, epochs=100, batch_size=64)
    
    # Predictions and evaluation
    test_predictions = predictor.predict(X_test)
    test_metrics = predictor.calculate_metrics(
        predictor.scaler.inverse_transform(y_test.reshape(-1, 1)),
        test_predictions
    )
    
    print("\nTest Set Performance:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save with custom name
    predictor.save_model('google_stock_model.h5')


def example_multiple_stocks():
    """
    Example: Compare predictions for multiple stocks.
    """
    print("\n" + "="*80)
    print("Example 3: Multiple Stock Comparison")
    print("="*80)
    
    stocks = ['AAPL', 'MSFT', 'GOOGL']
    results = {}
    
    for ticker in stocks:
        print(f"\nProcessing {ticker}...")
        
        predictor = LSTMStockPredictor(ticker=ticker, lookback_period=60)
        predictor.fetch_data(start_date='2020-01-01', end_date='2023-12-31')
        
        X_train, y_train, X_test, y_test, train_size = predictor.prepare_data(test_size=0.2)
        
        predictor.build_model(units=[50, 50], dropout=0.2)
        predictor.train(X_train, y_train, epochs=30, batch_size=32)
        
        test_predictions = predictor.predict(X_test)
        test_metrics = predictor.calculate_metrics(
            predictor.scaler.inverse_transform(y_test.reshape(-1, 1)),
            test_predictions
        )
        
        results[ticker] = test_metrics
        
        # Save visualization with ticker name
        predictor.visualize_predictions(
            y_train, y_test, 
            predictor.predict(X_train), test_predictions,
            train_size, save_path=f'{ticker}_prediction.png'
        )
    
    # Compare results
    print("\n" + "="*80)
    print("Comparison Results")
    print("="*80)
    for ticker, metrics in results.items():
        print(f"\n{ticker}:")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  R² Score: {metrics['R2 Score']:.4f}")


def example_load_and_predict():
    """
    Example: Load a saved model and make new predictions.
    """
    print("\n" + "="*80)
    print("Example 4: Load Saved Model")
    print("="*80)
    
    # Create new predictor instance
    predictor = LSTMStockPredictor(ticker='AAPL', lookback_period=60)
    
    # Load the saved model
    predictor.load_model('apple_stock_model.h5')
    
    # Fetch fresh data
    predictor.fetch_data()
    
    # Prepare data (scaler needs to be fitted)
    X_train, y_train, X_test, y_test, train_size = predictor.prepare_data(test_size=0.2)
    
    # Make predictions with loaded model
    predictions = predictor.predict(X_test)
    
    # Evaluate
    test_metrics = predictor.calculate_metrics(
        predictor.scaler.inverse_transform(y_test.reshape(-1, 1)),
        predictions
    )
    
    print("\nPerformance with loaded model:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    print("LSTM Stock Price Prediction - Usage Examples")
    print("=" * 80)
    print("\nNote: These examples require internet connection to fetch stock data.")
    print("Choose an example to run:\n")
    print("1. Basic usage with default settings")
    print("2. Custom configuration")
    print("3. Multiple stock comparison")
    print("4. Load and use saved model")
    print("\nTo run a specific example, uncomment the corresponding line below:\n")
    
    # Uncomment one of these to run:
    # example_basic_usage()
    # example_custom_configuration()
    # example_multiple_stocks()
    # example_load_and_predict()
    
    print("\nTo use these examples:")
    print("1. Edit this file and uncomment the example you want to run")
    print("2. Run: python examples.py")
    print("\nOr import and call the functions directly in your own script.")
