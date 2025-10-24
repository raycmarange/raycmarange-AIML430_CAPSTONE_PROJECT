# Usage Guide - NZX 50 Forecasting Pipeline

## Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/raycmarange/raycmarange-AIML430_CAPSTONE_PROJECT.git
cd raycmarange-AIML430_CAPSTONE_PROJECT
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Running the Pipeline

### Option 1: Python Script (Automated)

Run the complete pipeline with default settings:

```bash
python main.py
```

This will:
- Generate NZX 50 synthetic data (2020-2025)
- Engineer 30+ technical features
- Train LSTM and XGBoost models
- Generate ensemble predictions
- Perform risk analysis
- Create SHAP explanations
- Generate 5 visualization plots

**Output Files:**
- `predictions_plot.png` - Predictions vs Actual values
- `feature_importance.png` - Top features ranked by importance
- `shap_summary.png` - SHAP value analysis
- `training_loss.png` - LSTM training progress
- `risk_metrics.png` - Risk distribution and returns

### Option 2: Jupyter Notebook (Interactive)

For step-by-step exploration:

```bash
jupyter notebook main.ipynb
```

Then run cells sequentially to:
1. Import dependencies
2. Generate data
3. Engineer features
4. Prepare datasets
5. Train models
6. Make predictions
7. Analyze risks
8. Visualize results

## Customization

### Modifying Data Parameters

```python
# In main.py or notebook
data_gen = NZX50DataGenerator(
    start_date='2018-01-01',  # Earlier start
    end_date='2025-12-31'      # Later end
)
df_raw = data_gen.generate_data()
```

### Adjusting Model Parameters

#### LSTM Configuration
```python
forecaster = NZX50Forecaster(sequence_length=60)  # Longer sequences
lstm_history = forecaster.train_lstm(
    X_train_seq, 
    y_train_seq, 
    epochs=100,      # More training
    batch_size=64    # Larger batches
)
```

#### XGBoost Configuration
Modify in `main.py` line ~430:
```python
self.xgb_model = xgb.XGBRegressor(
    n_estimators=300,      # More trees
    max_depth=10,          # Deeper trees
    learning_rate=0.01,    # Slower learning
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42
)
```

### Ensemble Weights

Adjust model weighting:
```python
# Equal weights (default)
ensemble_pred = forecaster.ensemble_predict(
    X_test_seq, X_test_flat, 
    weights=[0.5, 0.5]
)

# Favor LSTM
ensemble_pred = forecaster.ensemble_predict(
    X_test_seq, X_test_flat, 
    weights=[0.7, 0.3]  # 70% LSTM, 30% XGBoost
)

# Favor XGBoost
ensemble_pred = forecaster.ensemble_predict(
    X_test_seq, X_test_flat, 
    weights=[0.3, 0.7]  # 30% LSTM, 70% XGBoost
)
```

## Using with Real Data

To integrate real market data, replace the `NZX50DataGenerator` with API calls:

```python
import yfinance as yf

# Example: Fetch NZX 50 data from Yahoo Finance
# Note: Use actual NZX 50 ticker or component stocks
ticker = "^NZ50"  # NZX 50 index (verify correct ticker)
df_raw = yf.download(ticker, start='2020-01-01', end='2025-10-01')

# Continue with feature engineering
feature_eng = FeatureEngineering()
df_features = feature_eng.add_technical_indicators(df_raw)
# ... rest of pipeline
```

## Understanding the Output

### Performance Metrics

**MSE (Mean Squared Error)**: Lower is better. Measures average squared difference.

**RMSE (Root Mean Squared Error)**: Lower is better. In same units as target.

**MAE (Mean Absolute Error)**: Lower is better. Average absolute error.

**R² (Coefficient of Determination)**: Higher is better (max 1.0). Variance explained.

**Directional Accuracy**: Percentage of correct directional predictions.

### Risk Metrics

**VaR (Value at Risk) 95%**: Maximum loss expected 95% of the time.

**CVaR (Conditional VaR) 95%**: Average loss when it exceeds VaR.

**Sharpe Ratio**: Risk-adjusted returns. Higher is better. >1.0 is good, >2.0 is excellent.

### SHAP Values

- **Red**: High feature value
- **Blue**: Low feature value
- **X-axis**: Impact on prediction
- Features at top: Most important

## Troubleshooting

### Out of Memory
Reduce batch size or sequence length:
```python
forecaster = NZX50Forecaster(sequence_length=15)
lstm_history = forecaster.train_lstm(X_train_seq, y_train_seq, batch_size=16)
```

### Slow Training
Reduce epochs or model complexity:
```python
lstm_history = forecaster.train_lstm(X_train_seq, y_train_seq, epochs=20)
```

### Import Errors
Ensure all dependencies installed:
```bash
pip install -r requirements.txt --upgrade
```

### CUDA/GPU Issues
Force CPU usage:
```python
# At top of main.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

## Advanced Usage

### Saving Models
```python
import joblib
import torch

# Save XGBoost model
joblib.dump(forecaster.xgb_model, 'xgb_model.pkl')

# Save LSTM model
torch.save(forecaster.lstm_model.state_dict(), 'lstm_model.pth')

# Save scalers
joblib.dump(forecaster.scaler, 'feature_scaler.pkl')
joblib.dump(forecaster.target_scaler, 'target_scaler.pkl')
```

### Loading Models
```python
# Load XGBoost
forecaster.xgb_model = joblib.load('xgb_model.pkl')

# Load LSTM
forecaster.lstm_model = LSTMModel(input_size=30)
forecaster.lstm_model.load_state_dict(torch.load('lstm_model.pth'))
forecaster.lstm_model.eval()

# Load scalers
forecaster.scaler = joblib.load('feature_scaler.pkl')
forecaster.target_scaler = joblib.load('target_scaler.pkl')
```

### Making Future Predictions
```python
# Prepare new data
new_data = feature_eng.add_technical_indicators(new_raw_data)
X_new = forecaster.scaler.transform(new_data.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1))

# Create sequences
X_new_seq = X_new[-30:].reshape(1, 30, -1)

# Predict
prediction = forecaster.predict_lstm(X_new_seq)
print(f"Next day prediction: {prediction[0][0]:.2f}")
```

## Performance Benchmarks

Typical execution times (on modern CPU):
- Data generation: < 1 second
- Feature engineering: < 5 seconds
- LSTM training (50 epochs): 2-5 minutes
- XGBoost training: 10-30 seconds
- Predictions: < 1 second
- SHAP analysis: 10-30 seconds
- Visualizations: 5-10 seconds

**Total pipeline: ~5-8 minutes**

## Contributing

To extend the pipeline:

1. Add new features in `FeatureEngineering` class
2. Add new models by creating model classes
3. Add new risk metrics in `RiskAnalyzer` class
4. Add new visualizations in `Visualizer` class

## Support

For issues or questions:
- Check documentation: README.md
- Review code comments in main.py
- Test with basic_test.py

---

**Ray Marange - AIML430 Capstone Project**  
*October 2025*
