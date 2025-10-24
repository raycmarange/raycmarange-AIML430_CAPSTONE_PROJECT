# NZX 50 Forecasting Pipeline with Enhanced XAI & Performance

**AI-Powered Stock Market Forecasting and Risk Analysis**

*Ray Marange - 09 October 2025*

## Overview

This project implements a comprehensive stock market forecasting pipeline for the NZX 50 index, combining state-of-the-art machine learning models with robust Explainable AI (XAI) features and advanced risk analysis capabilities.

### Key Features

- 🤖 **Multi-Model Ensemble**: Combines LSTM Neural Networks and XGBoost for robust predictions
- 📊 **Advanced Feature Engineering**: 30+ technical indicators including MA, EMA, MACD, RSI, Bollinger Bands
- 🔍 **Explainable AI**: SHAP values for model interpretability and transparency
- 📈 **Risk Analysis**: VaR, CVaR, and Sharpe Ratio calculations
- ⚡ **Performance Optimized**: Efficient data processing and model training
- 📉 **Comprehensive Visualizations**: Multiple plots for predictions, features, and risk metrics

## Dependencies

This project requires the following Python packages:

```
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
xgboost>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
shap>=0.42.0
sympy>=1.12
yfinance>=0.2.28
ta>=0.11.0
joblib>=1.3.0
tqdm>=4.66.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/raycmarange/raycmarange-AIML430_CAPSTONE_PROJECT.git
cd raycmarange-AIML430_CAPSTONE_PROJECT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Python Script

Run the complete pipeline:
```bash
python main.py
```

### Jupyter Notebook

For interactive exploration:
```bash
jupyter notebook main.ipynb
```

## Pipeline Architecture

### 1. Data Generation/Collection
- Generates synthetic NZX 50 index data with realistic patterns
- Includes trend, seasonality, random walk, and volatility components
- Extensible to real market data via APIs

### 2. Feature Engineering
Technical indicators computed:
- Moving Averages (MA): 5, 10, 20, 50 periods
- Exponential Moving Averages (EMA): 12, 26 periods
- MACD and Signal lines
- Relative Strength Index (RSI)
- Bollinger Bands
- Volatility metrics
- Momentum and Rate of Change (ROC)
- Volume indicators
- Lag features (1, 2, 3, 5, 7 periods)

### 3. Model Training

#### LSTM Neural Network
- 2-layer LSTM architecture
- 128 hidden units per layer
- Dropout regularization (0.2)
- Sequence-based time series modeling
- Adam optimizer with MSE loss

#### XGBoost Regressor
- 200 estimators
- Max depth: 7
- Learning rate: 0.05
- Subsample: 0.8
- Feature importance extraction

#### Ensemble Approach
- Weighted average of LSTM and XGBoost predictions
- Configurable weights (default: 50-50)

### 4. Risk Analysis
Comprehensive risk metrics:
- **Value at Risk (VaR)**: 95% confidence level
- **Conditional VaR (CVaR)**: Expected shortfall
- **Sharpe Ratio**: Risk-adjusted returns
- **Directional Accuracy**: Prediction direction correctness

### 5. Explainable AI (XAI)
- SHAP (SHapley Additive exPlanations) analysis
- Feature importance visualization
- Model interpretability insights
- Individual prediction explanations

### 6. Visualizations
Generated plots:
- Predictions vs Actual values
- Feature importance ranking
- SHAP summary plots
- Training loss curves
- Risk distribution and cumulative returns

## Project Structure

```
raycmarange-AIML430_CAPSTONE_PROJECT/
├── main.py              # Main Python script
├── main.ipynb           # Jupyter notebook version
├── requirements.txt     # Project dependencies
├── README.md           # This file
├── .gitignore          # Git ignore rules
└── outputs/            # Generated plots (created on run)
    ├── predictions_plot.png
    ├── feature_importance.png
    ├── shap_summary.png
    ├── training_loss.png
    └── risk_metrics.png
```

## Performance Metrics

The pipeline evaluates models using:
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **Directional Accuracy**: % of correct direction predictions

## Technical Details

### Classes and Modules

1. **NZX50DataGenerator**: Synthetic data generation with realistic market patterns
2. **FeatureEngineering**: Technical indicator calculation
3. **LSTMModel**: PyTorch LSTM neural network
4. **NZX50Forecaster**: Main forecasting class with ensemble methods
5. **RiskAnalyzer**: Risk metrics calculation
6. **XAIAnalyzer**: SHAP-based explainability
7. **Visualizer**: Plotting and visualization utilities

### Data Flow

```
Raw Data → Feature Engineering → Normalization → 
Sequence Creation → Model Training → Predictions → 
Risk Analysis → XAI Analysis → Visualizations
```

## Future Enhancements

- [ ] Integration with live market data APIs
- [ ] Additional models (Transformer, GRU)
- [ ] Hyperparameter optimization
- [ ] Real-time prediction dashboard
- [ ] Portfolio optimization module
- [ ] Sentiment analysis integration
- [ ] Multi-asset forecasting

## References

- XGBoost: Chen & Guestrin (2016)
- LSTM: Hochreiter & Schmidhuber (1997)
- SHAP: Lundberg & Lee (2017)
- Technical Analysis Library (ta)

## License

This project is part of AIML430 Capstone Project.

## Author

**Ray Marange**  
Applications and Implications of Artificial Intelligence  
October 2025

---

*Enhanced version with robust XAI and performance optimization*
