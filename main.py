#!/usr/bin/env python3
"""
NZX 50 Forecasting Pipeline with Enhanced XAI & Performance
AI-Powered Stock Market Forecasting and Risk Analysis
Ray Marange - 09 October 2025

Enhanced version with robust XAI and performance optimization
Dependencies: torch, pandas, numpy, xgboost, sklearn, matplotlib, seaborn, shap, sympy

Note: Ensure all dependencies are installed in your Python environment
Usage: python main.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import shap
import sympy as sp
from typing import Tuple, Dict, List
import joblib
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class NZX50DataGenerator:
    """
    Simulates NZX 50 stock market data for demonstration purposes.
    In production, this would fetch real data from financial APIs.
    """
    
    def __init__(self, start_date: str = '2020-01-01', end_date: str = '2025-10-01'):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
    def generate_data(self) -> pd.DataFrame:
        """Generate synthetic NZX 50 index data with realistic patterns."""
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        n = len(dates)
        
        # Generate base trend
        trend = np.linspace(10000, 12000, n)
        
        # Add seasonality
        seasonality = 500 * np.sin(2 * np.pi * np.arange(n) / 365.25)
        
        # Add random walk component
        random_walk = np.cumsum(np.random.randn(n) * 50)
        
        # Add volatility clustering
        volatility = np.abs(np.random.randn(n) * 100)
        
        # Combine components
        close = trend + seasonality + random_walk + volatility
        
        # Generate OHLC data
        df = pd.DataFrame({
            'Date': dates,
            'Open': close * (1 + np.random.uniform(-0.01, 0.01, n)),
            'High': close * (1 + np.random.uniform(0.005, 0.02, n)),
            'Low': close * (1 - np.random.uniform(0.005, 0.02, n)),
            'Close': close,
            'Volume': np.random.uniform(1e6, 5e6, n)
        })
        
        df.set_index('Date', inplace=True)
        return df


class FeatureEngineering:
    """
    Advanced feature engineering for stock market forecasting.
    """
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset."""
        df = df.copy()
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential moving averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # Price momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(4)
        
        # Rate of change
        df['ROC'] = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12)) * 100
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Lag features
        for lag in [1, 2, 3, 5, 7]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Return_Lag_{lag}'] = df['Close'].pct_change(lag)
        
        return df.dropna()


class LSTMModel(nn.Module):
    """
    LSTM Neural Network for time series forecasting.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        out = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out


class NZX50Forecaster:
    """
    Main forecasting class combining multiple models and XAI features.
    """
    
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.lstm_model = None
        self.xgb_model = None
        self.feature_names = None
        
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series modeling."""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
            
        return np.array(X_seq), np.array(y_seq)
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple:
        """Prepare data for training."""
        # Separate features and target
        target = df['Close'].values.reshape(-1, 1)
        features = df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
        
        self.feature_names = features.columns.tolist()
        
        # Scale features and target
        X_scaled = self.scaler.fit_transform(features)
        y_scaled = self.target_scaler.fit_transform(target)
        
        # Create sequences for LSTM
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        
        # Split data (80-20 split)
        split_idx = int(len(X_seq) * 0.8)
        
        X_train_seq = X_seq[:split_idx]
        y_train_seq = y_seq[:split_idx]
        X_test_seq = X_seq[split_idx:]
        y_test_seq = y_seq[split_idx:]
        
        # For XGBoost, use flat features (last sequence only)
        X_train_flat = X_scaled[self.sequence_length:split_idx + self.sequence_length]
        y_train_flat = y_scaled[self.sequence_length:split_idx + self.sequence_length]
        X_test_flat = X_scaled[split_idx + self.sequence_length:]
        y_test_flat = y_scaled[split_idx + self.sequence_length:]
        
        return (X_train_seq, y_train_seq, X_test_seq, y_test_seq,
                X_train_flat, y_train_flat, X_test_flat, y_test_flat)
    
    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray, 
                   epochs: int = 50, batch_size: int = 32) -> Dict:
        """Train LSTM model."""
        print("\n" + "="*60)
        print("Training LSTM Model")
        print("="*60)
        
        input_size = X_train.shape[2]
        self.lstm_model = LSTMModel(input_size)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=0.001)
        
        # Training loop
        train_losses = []
        
        for epoch in tqdm(range(epochs), desc="LSTM Training"):
            self.lstm_model.train()
            
            # Mini-batch training
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                # Forward pass
                outputs = self.lstm_model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            train_losses.append(loss.item())
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
        
        return {'train_losses': train_losses}
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Train XGBoost model."""
        print("\n" + "="*60)
        print("Training XGBoost Model")
        print("="*60)
        
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        self.xgb_model.fit(X_train, y_train.ravel(),
                          eval_set=[(X_train, y_train.ravel())],
                          verbose=False)
        
        print("XGBoost training completed!")
        
        return {'feature_importance': self.xgb_model.feature_importances_}
    
    def predict_lstm(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using LSTM model."""
        self.lstm_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions = self.lstm_model(X_tensor).numpy()
        return self.target_scaler.inverse_transform(predictions)
    
    def predict_xgboost(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using XGBoost model."""
        predictions = self.xgb_model.predict(X)
        return self.target_scaler.inverse_transform(predictions.reshape(-1, 1))
    
    def ensemble_predict(self, X_seq: np.ndarray, X_flat: np.ndarray, 
                        weights: List[float] = [0.5, 0.5]) -> np.ndarray:
        """Ensemble predictions from both models."""
        lstm_pred = self.predict_lstm(X_seq)
        xgb_pred = self.predict_xgboost(X_flat)
        
        # Weighted average
        ensemble_pred = weights[0] * lstm_pred + weights[1] * xgb_pred
        return ensemble_pred


class RiskAnalyzer:
    """
    Risk analysis and portfolio metrics calculation.
    """
    
    @staticmethod
    def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict:
        """Calculate performance metrics."""
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        
        # Financial metrics
        returns_actual = np.diff(actual.flatten()) / actual[:-1].flatten()
        returns_pred = np.diff(predicted.flatten()) / predicted[:-1].flatten()
        
        directional_accuracy = np.mean(np.sign(returns_actual) == np.sign(returns_pred)) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Directional_Accuracy': directional_accuracy
        }
    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk (VaR)."""
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        var = RiskAnalyzer.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe Ratio."""
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


class XAIAnalyzer:
    """
    Explainable AI analysis using SHAP values.
    """
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    def analyze(self, X: np.ndarray, sample_size: int = 100):
        """Perform SHAP analysis."""
        print("\n" + "="*60)
        print("Performing XAI Analysis with SHAP")
        print("="*60)
        
        # Sample data for efficiency
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Create SHAP explainer
        self.explainer = shap.Explainer(self.model.predict, X_sample)
        self.shap_values = self.explainer(X_sample)
        
        print("SHAP analysis completed!")
        
        return self.shap_values


class Visualizer:
    """
    Visualization utilities for the forecasting pipeline.
    """
    
    @staticmethod
    def plot_predictions(actual: np.ndarray, predicted: np.ndarray, 
                        title: str = "Predictions vs Actual"):
        """Plot predictions against actual values."""
        plt.figure(figsize=(14, 6))
        plt.plot(actual, label='Actual', linewidth=2, alpha=0.8)
        plt.plot(predicted, label='Predicted', linewidth=2, alpha=0.8)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('NZX 50 Index', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('predictions_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_feature_importance(importance: np.ndarray, feature_names: List[str], 
                               top_n: int = 15):
        """Plot feature importance."""
        indices = np.argsort(importance)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title('Top Feature Importance (XGBoost)', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_shap_summary(shap_values, feature_names: List[str]):
        """Plot SHAP summary."""
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        plt.title('SHAP Summary Plot - Feature Impact on Predictions', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_training_loss(losses: List[float]):
        """Plot training loss over epochs."""
        plt.figure(figsize=(10, 6))
        plt.plot(losses, linewidth=2)
        plt.title('LSTM Training Loss Over Epochs', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_risk_metrics(returns: np.ndarray):
        """Plot risk distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Returns distribution
        axes[0].hist(returns, bins=50, alpha=0.7, edgecolor='black')
        axes[0].axvline(returns.mean(), color='red', linestyle='--', 
                       linewidth=2, label='Mean')
        axes[0].set_title('Returns Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Returns', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        axes[1].plot(cumulative_returns, linewidth=2)
        axes[1].set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Time', fontsize=12)
        axes[1].set_ylabel('Cumulative Return', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('risk_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """
    Main execution function for the NZX 50 forecasting pipeline.
    """
    print("="*70)
    print(" " * 10 + "NZX 50 FORECASTING PIPELINE")
    print(" " * 5 + "AI-Powered Stock Market Forecasting and Risk Analysis")
    print(" " * 20 + "Ray Marange - 2025")
    print("="*70)
    print("\nEnhanced version with robust XAI and performance optimization\n")
    
    # Step 1: Generate/Load Data
    print("\n[1/7] Generating NZX 50 Market Data...")
    data_gen = NZX50DataGenerator(start_date='2020-01-01', end_date='2025-10-01')
    df_raw = data_gen.generate_data()
    print(f"✓ Generated {len(df_raw)} days of market data")
    print(f"   Date range: {df_raw.index[0]} to {df_raw.index[-1]}")
    
    # Step 2: Feature Engineering
    print("\n[2/7] Engineering Features...")
    feature_eng = FeatureEngineering()
    df_features = feature_eng.add_technical_indicators(df_raw)
    print(f"✓ Created {len(df_features.columns)} features")
    print(f"   Sample features: {', '.join(df_features.columns[:5].tolist())}...")
    
    # Step 3: Prepare Data
    print("\n[3/7] Preparing Data for Training...")
    forecaster = NZX50Forecaster(sequence_length=30)
    (X_train_seq, y_train_seq, X_test_seq, y_test_seq,
     X_train_flat, y_train_flat, X_test_flat, y_test_flat) = forecaster.prepare_data(df_features)
    
    print(f"✓ Training sequences: {X_train_seq.shape}")
    print(f"✓ Testing sequences: {X_test_seq.shape}")
    
    # Step 4: Train Models
    print("\n[4/7] Training Forecasting Models...")
    
    # Train LSTM
    lstm_history = forecaster.train_lstm(X_train_seq, y_train_seq, epochs=50, batch_size=32)
    
    # Train XGBoost
    xgb_history = forecaster.train_xgboost(X_train_flat, y_train_flat)
    
    # Step 5: Make Predictions
    print("\n[5/7] Generating Predictions...")
    
    # Ensemble predictions
    ensemble_pred = forecaster.ensemble_predict(X_test_seq, X_test_flat, weights=[0.5, 0.5])
    y_test_actual = forecaster.target_scaler.inverse_transform(y_test_seq)
    
    # Individual model predictions
    lstm_pred = forecaster.predict_lstm(X_test_seq)
    xgb_pred = forecaster.predict_xgboost(X_test_flat)
    
    print("✓ Predictions generated successfully")
    
    # Step 6: Risk Analysis
    print("\n[6/7] Performing Risk Analysis...")
    risk_analyzer = RiskAnalyzer()
    
    # Calculate metrics
    ensemble_metrics = risk_analyzer.calculate_metrics(y_test_actual, ensemble_pred)
    lstm_metrics = risk_analyzer.calculate_metrics(y_test_actual, lstm_pred)
    xgb_metrics = risk_analyzer.calculate_metrics(y_test_actual, xgb_pred)
    
    # Calculate risk metrics
    returns = np.diff(y_test_actual.flatten()) / y_test_actual[:-1].flatten()
    var_95 = risk_analyzer.calculate_var(returns, 0.95)
    cvar_95 = risk_analyzer.calculate_cvar(returns, 0.95)
    sharpe = risk_analyzer.calculate_sharpe_ratio(returns)
    
    # Print metrics
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    
    print("\nEnsemble Model:")
    for metric, value in ensemble_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nLSTM Model:")
    for metric, value in lstm_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nXGBoost Model:")
    for metric, value in xgb_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n" + "="*60)
    print("RISK METRICS")
    print("="*60)
    print(f"  Value at Risk (95%): {var_95:.4f}")
    print(f"  Conditional VaR (95%): {cvar_95:.4f}")
    print(f"  Sharpe Ratio: {sharpe:.4f}")
    
    # Step 7: XAI Analysis
    print("\n[7/7] Performing Explainable AI Analysis...")
    xai_analyzer = XAIAnalyzer(forecaster.xgb_model, forecaster.feature_names)
    shap_values = xai_analyzer.analyze(X_test_flat, sample_size=100)
    
    # Visualization
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    visualizer = Visualizer()
    
    print("\n→ Plotting predictions vs actual...")
    visualizer.plot_predictions(y_test_actual, ensemble_pred, 
                               "NZX 50 Ensemble Model: Predictions vs Actual")
    
    print("→ Plotting feature importance...")
    visualizer.plot_feature_importance(xgb_history['feature_importance'], 
                                      forecaster.feature_names)
    
    print("→ Plotting SHAP summary...")
    visualizer.plot_shap_summary(shap_values, forecaster.feature_names)
    
    print("→ Plotting training loss...")
    visualizer.plot_training_loss(lstm_history['train_losses'])
    
    print("→ Plotting risk metrics...")
    visualizer.plot_risk_metrics(returns)
    
    print("\n" + "="*70)
    print(" " * 20 + "PIPELINE COMPLETED")
    print("="*70)
    print("\n✓ All visualizations saved to current directory:")
    print("  - predictions_plot.png")
    print("  - feature_importance.png")
    print("  - shap_summary.png")
    print("  - training_loss.png")
    print("  - risk_metrics.png")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
