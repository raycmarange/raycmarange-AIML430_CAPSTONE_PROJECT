# =============================================================================
# COMPLETE DATA PIPELINE FOR FINANCIAL TIME SERIES - ENHANCED PERFORMANCE VERSION
# =============================================================================
# Add to imports section
from symbolic_xai import SymbolicMarketRegressor, EnhancedXAIAnalysis
import pandas as pd
import numpy as np
import yfinance as yf
from torch.utils.data import Dataset, DataLoader
import torch
from config import DataConfig, TrainingConfig, ForecastConfig
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mstats # Requires scipy
import warnings
warnings.filterwarnings('ignore')

class ForecastingDataset(Dataset):
    """PyTorch Dataset for forecasting sequences"""
    
    def __init__(self, sequences, targets_6m, targets_1m, regimes):
        self.sequences = sequences
        self.targets_6m = targets_6m
        self.targets_1m = targets_1m
        self.regimes = regimes
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
            sequence = torch.FloatTensor(self.sequences[idx])
            
            # Handle both regression and classification targets
            target_6m_reg = torch.FloatTensor([self.targets_6m[idx][0]])
            target_6m_cls = torch.LongTensor([self.targets_6m[idx][1]]).squeeze() # Squeeze for CrossEntropyLoss
            
            target_1m_reg = torch.FloatTensor([self.targets_1m[idx][0]])
            target_1m_cls = torch.LongTensor([self.targets_1m[idx][1]]).squeeze() # Squeeze for CrossEntropyLoss
            
            regime = torch.FloatTensor([self.regimes[idx]])
            
            # Must return a tuple/list for positional unpacking in the trainer
            # Order must match trainer's unpack_batch logic: 
            # features, reg_target_6m, class_target_6m, reg_target_1m, class_target_1m, regime
            return (
                sequence, 
                target_6m_reg, 
                target_6m_cls, 
                target_1m_reg, 
                target_1m_cls, 
                regime
            )
class EnhancedCrisisDetector:
    """Enhanced crisis detection with advanced regime analysis"""
    
    NZ_CRISIS_PERIODS = {
        'Global Financial Crisis': ('2007-10-01', '2009-03-01'),
        'European Debt Crisis': ('2010-04-01', '2012-07-01'), 
        'COVID-19 Pandemic': ('2020-02-20', '2020-04-30'),
        '2022 Inflation/Ukraine': ('2022-01-01', '2022-12-31'),
        'NZ Housing Correction 2018': ('2018-01-01', '2018-12-31')
    }
    
    KNOWN_CRISES = {
        'Dot-com Bubble': ('2000-03-01', '2002-10-01'),
        'Global Financial Crisis': ('2007-10-01', '2009-03-01'),
        'European Debt Crisis': ('2010-04-01', '2012-07-01'),
        'COVID-19 Pandemic': ('2020-02-01', '2020-04-01'),
        '2022 Inflation Crisis': ('2022-01-01', '2022-12-01')
    }
    
    @staticmethod
    def detect_volatility_regimes(data, window=30, threshold=1.5):
        if 'Close' not in data.columns:
            return pd.Series([0] * len(data), index=data.index)
            
        returns = data['Close'].pct_change()
        rolling_vol = returns.rolling(window).std()
        vol_threshold = rolling_vol.median() * threshold
        
        high_vol_periods = (rolling_vol > vol_threshold).astype(int)
        return high_vol_periods.fillna(0)
    
    @staticmethod
    def detect_drawdown_regimes(data, threshold=0.10):
        if 'Close' not in data.columns:
            return pd.Series([0] * len(data), index=data.index)
            
        cumulative = (1 + data['Close'].pct_change()).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        stress_periods = (drawdown < -threshold).astype(int)
        return stress_periods.fillna(0)
    
    @staticmethod
    def label_historical_crises(data_index):
        crisis_labels = pd.Series(0, index=data_index, name='historical_crisis')
        
        for crisis_name, (start, end) in EnhancedCrisisDetector.NZ_CRISIS_PERIODS.items():
            mask = (data_index >= start) & (data_index <= end)
            if mask.any():
                crisis_labels[mask] = 1
                print(f"📍 {crisis_name}: {mask.sum()} trading days")
                
        return crisis_labels

    def analyze_market_regimes(self, data):
        print("\n🔍 ENHANCED MARKET REGIME ANALYSIS")
        print("=" * 50)
        
        # Multiple detection methods
        vol_stress = self.detect_volatility_regimes(data)
        drawdown_stress = self.detect_drawdown_regimes(data)
        historical_crises = self.label_historical_crises(data.index)
        
        # Advanced regime scoring
        regime_score = (
            vol_stress.astype(float) * 0.4 +
            drawdown_stress.astype(float) * 0.4 + 
            historical_crises.astype(float) * 0.2
        )
        
        # Adaptive threshold
        threshold = regime_score.quantile(0.75)
        combined_stress = (regime_score > threshold).astype(int)
        
        print(f"📈 High Volatility Periods: {vol_stress.sum()} days")
        print(f"📉 Significant Drawdowns: {drawdown_stress.sum()} days")
        print(f"📚 Historical Crisis Days: {historical_crises.sum()} days")
        print(f"🎯 Combined Stress Periods: {combined_stress.sum()} days")
        print(f"📊 Stress Ratio: {combined_stress.mean():.1%}")
        print(f"📈 Regime Score Stats - Mean: {regime_score.mean():.3f}, Max: {regime_score.max():.3f}")
        
        return {
            'volatility_stress': vol_stress,
            'drawdown_stress': drawdown_stress, 
            'historical_crises': historical_crises,
            'combined_stress': combined_stress,
            'regime_score': regime_score
        }

    def plot_crisis_timeline(self, data):
        crisis_data = self.analyze_market_regimes(data)
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(data.index, data['Close'], label='NZX 50 Price', linewidth=1, color='blue')
        
        crisis_mask = crisis_data['combined_stress'] == 1
        crisis_dates = data.index[crisis_mask]
        crisis_prices = data['Close'][crisis_mask]
        
        plt.scatter(crisis_dates, crisis_prices, color='red', s=10, alpha=0.6, 
                   label=f'Crisis Periods ({len(crisis_dates)} days)')
        
        plt.title('NZX 50 Price with Crisis Periods')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        returns = data['Close'].pct_change().dropna()
        
        normal_returns = returns[crisis_data['combined_stress'].iloc[1:] == 0]
        crisis_returns = returns[crisis_data['combined_stress'].iloc[1:] == 1]
        
        if len(normal_returns) > 0 and len(crisis_returns) > 0:
            plt.hist(normal_returns, bins=50, alpha=0.7, label=f'Normal ({len(normal_returns)} days)', color='green')
            plt.hist(crisis_returns, bins=50, alpha=0.7, label=f'Crisis ({len(crisis_returns)} days)', color='red')
        
        plt.title('Return Distribution: Normal vs Crisis Periods')
        plt.xlabel('Daily Returns')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return crisis_data

class FinancialDataPipeline:
    def __init__(self, config=DataConfig()):
        self.config = config
        self.crisis_detector = EnhancedCrisisDetector()
        self.nz50_ticker = None
        self.performance_metrics = {}
    
    def fetch_live_price(self, ticker=None):
        """
        Fetch live current price with fallback strategy
        Returns: current_price (float) or None if failed
        """
        import requests
        import json
        
        if ticker is None:
            ticker = self.nz50_ticker or self.config.TICKERS['nz_primary']
        
        print(f"📡 Attempting to fetch live price for {ticker}...")
        
        # Method 1: Yahoo Finance API
        try:
            yahoo_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            params = {'range': '1d', 'interval': '1m'}
            
            response = requests.get(yahoo_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'chart' in data and 'result' in data['chart']:
                    result = data['chart']['result'][0]
                    current_price = result['meta']['regularMarketPrice']
                    print(f"✅ Live price from Yahoo: ${current_price:.2f}")
                    return float(current_price)
        except Exception as e:
            print(f"❌ Yahoo Finance failed: {e}")
        
        # Method 2: yfinance fallback
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            current_data = stock.history(period='1d', interval='1m')
            if not current_data.empty:
                current_price = current_data['Close'].iloc[-1]
                print(f"✅ Live price from yfinance: ${current_price:.2f}")
                return float(current_price)
        except Exception as e:
            print(f"❌ yfinance fallback failed: {e}")
        
        # Method 3: Fallback to historical data
        try:
            if hasattr(self, 'market_data') and 'nz_primary' in self.market_data:
                historical_data = self.market_data['nz_primary']
                if not historical_data.empty:
                    current_price = historical_data['Close'].iloc[-1]
                    print(f"⚠️ Using latest historical price: ${current_price:.2f}")
                    return float(current_price)
        except Exception as e:
            print(f"❌ Historical fallback failed: {e}")
        
        print("❌ All price fetching methods failed")
        return None

    def detect_nzx50_ticker(self):
        """Automatically detect available NZX 50 ticker with enhanced fallback"""
        print("🔍 Detecting available NZX 50 ticker...")
        
        for ticker in self.config.NZ50_OPTIONS:
            success, data = self._download_single_ticker(ticker, "NZX 50 Probe")
            if success:
                print(f"✅ NZX 50 available: {ticker} ({len(data)} trading days)")
                self.nz50_ticker = ticker
                return ticker
        
        # Enhanced fallback strategy
        fallback_candidates = ['AIA.NZ', 'FPH.NZ', 'MRP.NZ', 'MEL.NZ']
        for candidate in fallback_candidates:
            success, data = self._download_single_ticker(candidate, f"Fallback {candidate}")
            if success:
                print(f"🔄 Using fallback constituent: {candidate}")
                self.nz50_ticker = candidate
                return candidate
        
        raise Exception("❌ No NZX 50 data or major constituents available")

    def fetch_market_data(self):
        """Fetch market data with enhanced error handling and performance tracking"""
        print("📊 Fetching comprehensive market data...")
        
        nz_ticker = self.detect_nzx50_ticker()
        self.config.TICKERS['nz_primary'] = nz_ticker
        
        market_data = {}
        used_tickers = {}
        failed_tickers = []
        
        for ticker_key, ticker in self.config.TICKERS.items():
            if ticker is None:
                continue
                
            success, data = self._download_single_ticker(ticker, ticker_key)
            if success:
                market_data[ticker_key] = data
                used_tickers[ticker_key] = ticker
                
                # Track data quality metrics
                self._track_data_quality(ticker_key, data)
            else:
                failed_tickers.append(ticker)
                print(f"⚠️  Failed to fetch {ticker_key}: {ticker}")
        
        if not market_data:
            raise Exception("❌ No market data could be fetched")
        
        print(f"✅ Successfully fetched {len(market_data)} datasets")
        for key, ticker in used_tickers.items():
            data_type = "NZX 50 Index" if key == 'nz_primary' and ticker.startswith('^') else "Stock/ETF"
            print(f"   • {key}: {ticker} ({data_type}) - {len(market_data[key])} days")
        
        if failed_tickers:
            print(f"⚠️  Failed to fetch {len(failed_tickers)} tickers: {failed_tickers}")
        
        return market_data

    def _track_data_quality(self, ticker_key, data):
        """Track data quality metrics for performance optimization"""
        quality_metrics = {
            'total_days': len(data),
            'missing_days': data.isnull().sum().sum(),
            'zero_volume_days': (data.get('Volume', pd.Series([0])).eq(0).sum() if 'Volume' in data else 0),
            'price_range': (data['Close'].max() - data['Close'].min()) / data['Close'].mean() if len(data) > 0 else 0,
            'volatility': data['Close'].pct_change().std() if len(data) > 1 else 0
        }
        self.performance_metrics[ticker_key] = quality_metrics

    def engineer_advanced_features(self, data):
        """Enhanced feature engineering with better outlier handling"""
        print("🔧 Engineering advanced features...")
        
        # Create base features
        df = self._engineer_robust_features(data)
        
        # Add advanced features
        df = self._add_advanced_technical_features(df)
        df = self._add_market_microstructure_features(df)
        
        # Apply robust preprocessing to unstable features
        df = self._apply_robust_feature_processing(df)
        # 🟢 CRITICAL FIX INVOCATION: Ensure this definitive fix runs last on illiquidity
        df = self._fix_amihud_critical(df) # <-- INVOKE THE NEW METHOD HERE
        df = self._add_regime_aware_features(df)
        
        # Enhanced crisis detection
        crisis_data = self.crisis_detector.analyze_market_regimes(data)
        df = self._integrate_regime_features(df, crisis_data)
        
        # Remove non-numeric regime category before feature selection
        if 'regime_category' in df.columns:
            df = df.drop(columns=['regime_category'])
        
        # Conservative feature selection - preserve important signals
        df = self._apply_conservative_feature_selection(df)
        
        # Final data cleaning
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        print(f"✅ Advanced feature engineering: {len(df)} samples, {len(df.columns)} features")

        return df
    
    def _handle_volume_outliers(self, data):
        """Implement robust outlier detection for Volume."""
        if 'Volume' not in data.columns:
            return data
            
        volume = data['Volume'].copy()
        
        # 1. Zero Volume Handling: Remove or impute extreme zero-volume days
        zero_volume_days = volume.eq(0).sum()
        if zero_volume_days / len(data) > 0.05:
            # Impute if too many
            print(f"⚠️ High zero volume ratio ({zero_volume_days/len(data):.2%}). Imputing with median.")
            volume[volume == 0] = volume[volume > 0].median()
        elif zero_volume_days > 0:
            print(f"🔄 Replacing {zero_volume_days} zero volume days with median.")
            volume[volume == 0] = volume[volume > 0].median()
        
        # 2. Winsorization (robust outlier detection)
        # Use log-transformation for robustness before winsorizing
        log_volume = np.log1p(volume)
        
        # Winsorize outliers (clip to 1st and 99th percentiles)
        lower_bound = log_volume.quantile(0.01)
        upper_bound = log_volume.quantile(0.99)
        
        log_volume_winsorized = np.clip(log_volume, lower_bound, upper_bound)
        
        # Replace the original volume with the winsorized log-volume (for use in log_volume feature)
        # We don't inverse transform it because we immediately take log in _engineer_robust_features
        data['Volume_Winsorized'] = np.expm1(log_volume_winsorized)
        
        print("✅ Applied robust log-winsorization for Volume outliers.")
        return data.drop(columns=['Volume']).rename(columns={'Volume_Winsorized': 'Volume'})

    def _fix_illiquidity_features(self, df):
        """Fixes amihud and volume volatility features with robust log-clipping and standardization."""
        
        if 'amihud_illiquidity' in df.columns:
            amihud = df['amihud_illiquidity'].copy()
            
            # 1. Handle NaNs/Infs/Zeroes, replacing with median for robust statistics
            amihud = amihud.replace([np.inf, -np.inf], np.nan)
            amihud = amihud.fillna(amihud.median())
            
            # 2. Apply log transformation (log1p handles values close to zero)
            amihud = np.log1p(np.abs(amihud))
            
            # 3. Winsorize outliers (clip to 1st and 99th percentiles)
            lower = amihud.quantile(0.01)
            upper = amihud.quantile(0.99)
            amihud = np.clip(amihud, lower, upper)
            
            # 4. Final Standardization (Necessary since the initial normalization failed)
            amihud = (amihud - amihud.mean()) / (amihud.std() + 1e-8)
            
            df['amihud_illiquidity'] = amihud
            print("🔄 Fixed amihud_illiquidity via log-winsorization and standardization.")

        # Apply similar fix to volume_volatility if needed
        if 'volume_volatility' in df.columns:
            vol_vol = df['volume_volatility'].copy()
            vol_vol = vol_vol.replace([np.inf, -np.inf], np.nan).fillna(vol_vol.median())
            
            # Log transform to compress large values
            vol_vol = np.log1p(np.abs(vol_vol))
            
            # Winsorize to prevent outliers
            lower = vol_vol.quantile(0.01)
            upper = vol_vol.quantile(0.99)
            vol_vol = np.clip(vol_vol, lower, upper)
            
            # Final Standardization
            vol_vol = (vol_vol - vol_vol.mean()) / (vol_vol.std() + 1e-8)
            
            df['volume_volatility'] = vol_vol
            print("🔄 Fixed volume_volatility via log transformation and standardization.")
            
        return df
    def _add_advanced_technical_features(self, df):
        """Add advanced technical indicators"""
        # Price-based features
        df['price_velocity'] = df['Close'].pct_change(5) - df['Close'].pct_change(20)
        df['price_acceleration'] = df['price_velocity'].diff(5)
        
        # Enhanced volatility features
        for short, long in [(5, 20), (10, 30)]:
            df[f'volatility_ratio_{short}_{long}'] = (
                df[f'volatility_{short}d'] / df[f'volatility_{long}d']
            )
        
        # Momentum combinations
        df['momentum_composite'] = (
            0.4 * df['momentum_1m'] + 
            0.3 * df['momentum_3m'] + 
            0.3 * df['momentum_6m']
        )
        
        # Trend strength
        df['trend_strength'] = df['Close'].rolling(20).apply(
            lambda x: abs(np.polyfit(range(len(x)), x, 1)[0]) / np.std(x), raw=True
        )
        
        return df

    def _add_market_microstructure_features(self, df):
        """Add market microstructure features"""
        if 'Volume' in df.columns:
            # Volume-based features
            df['volume_momentum'] = df['Volume'].pct_change(5)
            df['volume_volatility'] = df['Volume'].rolling(10).std()
            df['volume_price_correlation'] = df['Volume'].rolling(20).corr(df['Close'])
            
            # Liquidity measures
            df['amihud_illiquidity'] = (
                abs(df['returns']) / (df['Volume'] + 1e-8)
            ).rolling(10).mean()
        
        # Gap features
        df['overnight_gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['intraday_range'] = (df['High'] - df['Low']) / df['Open']
        
        return df

    def _add_regime_aware_features(self, df):
        """Add features that adapt to market regimes"""
        if 'stress_period' in df.columns:
            # Regime-adjusted technical indicators
            df['rsi_regime_adjusted'] = df['rsi_14'] * (1 + 0.3 * df['stress_period'])
            df['volatility_regime_adjusted'] = df['volatility_20d'] * (1 + 0.5 * df['stress_period'])
            
            # Regime-specific momentum
            df['momentum_stress_adjusted'] = df['momentum_1m'] * (1 - 0.2 * df['stress_period'])
        
        return df

    def _apply_feature_selection(self, df):
        """Apply feature selection to remove redundant features"""
        # 🆕 CRITICAL FIX: Remove non-numeric categorical columns before calculating correlation
        if 'regime_category' in df.columns:
            print("🔄 Dropping non-numeric 'regime_category' before correlation.")
            df = df.drop(columns=['regime_category'])
        # Remove highly correlated features
        correlation_matrix = df.corr().abs()
        upper_tri = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_tri.columns 
                  if any(upper_tri[column] > 0.95)]
        
        if to_drop:
            print(f"🔄 Removing highly correlated features: {to_drop}")
            df = df.drop(columns=to_drop)
        
        # Remove low variance features (protected features excluded)
        protected_features = ['stress_period', 'crisis_period', 'target_']
        variance_threshold = 0.01
        
        low_variance_features = []
        for column in df.columns:
            if any(prot in column for prot in protected_features):
                continue
            if df[column].var() < variance_threshold:
                low_variance_features.append(column)
        
        if low_variance_features:
            print(f"🔄 Removing low variance features: {low_variance_features}")
            df = df.drop(columns=low_variance_features)
        
        return df

    def _integrate_regime_features(self, df, crisis_data):
        """Integrate enhanced regime features"""
        for key, series in crisis_data.items():
            if key not in ['volatility_stress', 'drawdown_stress', 'historical_crises', 'combined_stress']:
                continue
            # Align the crisis data with the feature DataFrame
            aligned_series = series.reindex(df.index).fillna(0)
            df[f'crisis_{key}'] = aligned_series
        
        # Create composite regime score
        regime_columns = [col for col in df.columns if 'crisis_' in col or 'stress' in col]
        if regime_columns:
            df['composite_regime_score'] = df[regime_columns].mean(axis=1)
            df['regime_category'] = pd.cut(
                df['composite_regime_score'], 
                bins=[-0.1, 0.3, 0.7, 1.1], 
                labels=['calm', 'moderate', 'stress']
            )
        
        return df


    # Keep existing methods but ensure they use enhanced versions
    def prepare_forecast_data(self, market_data):
        """Prepare data using advanced feature engineering"""
        print("🎯 Preparing forecast data with advanced features...")
        
        nzx_key = self.config.TICKERS['nz_primary']
        if nzx_key not in market_data:
            nzx_key = list(market_data.keys())[0]
        
        # Use advanced features instead of robust features
        features_df = self.engineer_advanced_features(market_data[nzx_key])
        
        forecast_sequences = self._create_balanced_forecast_sequences(
            features_df, 
            TrainingConfig.SEQUENCE_LENGTH,
            ForecastConfig.FORECAST_PERIODS
        )
        
        return forecast_sequences, features_df

    def get_performance_report(self):
        """Generate performance optimization report"""
        report = {
            'data_quality': self.performance_metrics,
            'feature_engineering': {
                'advanced_features_added': [
                    'price_velocity', 'price_acceleration', 'volatility_ratios',
                    'momentum_composite', 'trend_strength', 'regime_adjusted_indicators'
                ],
                'optimization_strategies': [
                    'Correlation-based feature selection',
                    'Variance threshold filtering',
                    'Regime-aware feature engineering'
                ]
            },
            'cross_validation': {
                'strategy': 'Time-series aware with regime balancing',
                'benefits': 'More robust performance estimation'
            }
        }
        return report

    # Maintain existing methods for compatibility
    def _engineer_robust_features(self, price_data):
        """Enhanced version of robust features"""
        df = price_data.copy()
        
        # Ensure we're working with proper Series
        for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
            if col in df.columns and isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]
        
        # Basic returns and momentum
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Momentum features
        df['momentum_1m'] = df['Close'].pct_change(21)
        df['momentum_3m'] = df['Close'].pct_change(63)
        df['momentum_6m'] = df['Close'].pct_change(126)
        
        # Volatility features
        for window in [5, 10, 20, 30]:
            vol_col = f'volatility_{window}d'
            df[vol_col] = df['returns'].rolling(window).std()
            df[vol_col] = df[vol_col].bfill().ffill()
        
        # Technical indicators
        df['rsi_14'] = self._calculate_rsi(df['Close'], 14) / 100.0
        df['macd'], df['macd_signal'] = self._calculate_macd(df['Close'])
        
        # Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger_bands(df['Close'])
        df['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Volume features
        if 'Volume' in df.columns:
            # 🆕 FIX: Use log1p transformation for robust scaling validation
            df['log_volume'] = np.log1p(df['Volume'])
            volume_sma = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = (df['Volume'] / (volume_sma + 1e-8)) - 1.0
        
        # Targets
        df = self._add_multi_horizon_targets(df)
        
        # Regime features
        df = self._add_robust_regime_features(df)
        
        # Remove constant features
        df = self._remove_constant_features(df)
        
        df = df.dropna()
        return df

    def _add_multi_horizon_targets(self, features_df):
        """Fixed target calculation"""
        features_df['target_return_6m'] = features_df['Close'].shift(-126) / features_df['Close'] - 1
        features_df['target_direction_6m'] = (features_df['target_return_6m'] > 0).astype(int)
        
        features_df['target_return_1m'] = features_df['Close'].shift(-21) / features_df['Close'] - 1  
        features_df['target_direction_1m'] = (features_df['target_return_1m'] > 0).astype(int)
        
        # Clip extreme returns
        return_clip = 0.5
        features_df['target_return_6m'] = np.clip(features_df['target_return_6m'], -return_clip, return_clip)
        features_df['target_return_1m'] = np.clip(features_df['target_return_1m'], -return_clip, return_clip)
        
        return features_df

    def _create_balanced_forecast_sequences(self, features_df, sequence_length, forecast_horizon):
        """Create sequences with guaranteed regime balance and non-overlapping time splits."""
        feature_cols = [col for col in features_df.columns if not col.startswith('target_')]
        
        crisis_cols = ['crisis_period', 'stress_period', 'high_volatility']
        crisis_col = next((col for col in crisis_cols if col in features_df.columns), None)
        
        if crisis_col is None:
            features_df['regime'] = (features_df['returns'].rolling(30).std() > 
                                    features_df['returns'].rolling(30).std().quantile(0.7)).astype(int)
            crisis_col = 'regime'
        
        total_samples = len(features_df)
        
        # 🆕 FIX: Implement robust time-series split to prevent data leakage
        split_point_80 = int(0.7 * total_samples) # Train/Val split
        split_point_90 = int(0.85 * total_samples) # Val/Test split
        
        # Ensure splits account for sequence length and forecast horizon
        train_end_idx = split_point_80 - forecast_horizon - 1 
        val_end_idx = split_point_90 - forecast_horizon - 1
        
        if train_end_idx < sequence_length or val_end_idx < train_end_idx:
            print("⚠️ Data too short for strict time-series splits.")
        
        split_date_80 = features_df.index[max(sequence_length, train_end_idx)]
        split_date_90 = features_df.index[max(sequence_length, val_end_idx)]
        
        print(f"📅 Time-based splits: Train end date: {split_date_80}, Val end date: {split_date_90}")
        
        sequences = []
        targets_6m = []
        targets_1m = []
        regimes = []
        split_labels = []
        
        for i in range(sequence_length, len(features_df) - forecast_horizon):
            current_date = features_df.index[i]
            
            if current_date < split_date_80:
                split_label = 'train'
            elif current_date < split_date_90:
                split_label = 'val'
            else:
                split_label = 'test'
            
            try:
                seq = features_df[feature_cols].iloc[i-sequence_length:i].values
                target_6m_reg = features_df['target_return_6m'].iloc[i]
                target_6m_class = features_df['target_direction_6m'].iloc[i]
                target_1m_reg = features_df['target_return_1m'].iloc[i]
                target_1m_class = features_df['target_direction_1m'].iloc[i]
                regime = features_df[crisis_col].iloc[i]
                
                if (pd.isna(target_6m_reg) or pd.isna(target_1m_reg) or 
                    pd.isna(target_6m_class) or pd.isna(target_1m_class)):
                    continue
                
                sequences.append(seq)
                targets_6m.append((target_6m_reg, target_6m_class))
                targets_1m.append((target_1m_reg, target_1m_class))
                regimes.append(regime)
                split_labels.append(split_label)
                
            except (KeyError, IndexError) as e:
                continue
        
        sequences = np.array(sequences)
        targets_6m = np.array(targets_6m)
        targets_1m = np.array(targets_1m)
        regimes = np.array(regimes)
        split_labels = np.array(split_labels)
        
        sequence_df = pd.DataFrame({
            'sequence': list(sequences),
            'target_6m': list(targets_6m),
            'target_1m': list(targets_1m),
            'regime': regimes,
            'split': split_labels
        })
        
        # Balance test set - Keep existing logic for regime balance check
        test_sequences = sequence_df[sequence_df['split'] == 'test']
        test_stress_count = (test_sequences['regime'] == 1).sum()
        
        if test_stress_count == 0:
            print("🔄 Rebalancing test set to include stress periods...")
            val_sequences = sequence_df[sequence_df['split'] == 'val']
            val_stress = val_sequences[val_sequences['regime'] == 1]
            
            if len(val_stress) > 0:
                # Move a maximum of 10 stress samples
                move_count = min(10, len(val_stress))
                move_indices = val_stress.index[:move_count]
                sequence_df.loc[move_indices, 'split'] = 'test'
                print(f"📊 Moved {move_count} stress samples from validation to test set")
        
        final_test = sequence_df[sequence_df['split'] == 'test']
        final_stress_count = (final_test['regime'] == 1).sum()
        final_normal_count = (final_test['regime'] == 0).sum()
        
        print(f"✅ Final test set balance: {final_stress_count} stress, {final_normal_count} normal")
        
        result = {
            'sequences': np.array(list(sequence_df['sequence'])),
            'targets_6m': np.array(list(sequence_df['target_6m'])),
            'targets_1m': np.array(list(sequence_df['target_1m'])),
            'regimes': sequence_df['regime'].values,
            'split_labels': sequence_df['split'].values,
            'feature_names': feature_cols
        }
        
        return result

    # Technical indicator methods (keep existing)
    def _calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0] if len(prices.columns) > 0 else prices.squeeze()
        
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band
    
    def _apply_robust_feature_processing(self, df):
        """Apply robust processing to problematic features"""
        # Handle volume outliers more conservatively
        if 'Volume' in df.columns:
            # Use median-based winsorizing instead of log transformation
            volume_median = df['Volume'].median()
            volume_std = df['Volume'].std()
            upper_bound = volume_median + 3 * volume_std
            lower_bound = max(volume_median - 3 * volume_std, 0)
            
            df['Volume'] = np.clip(df['Volume'], lower_bound, upper_bound)
            print("🔄 Applied conservative winsorizing to Volume outliers")
        
        # Handle illiquidity features with robust scaling
        illiquidity_features = ['amihud_illiquidity', 'volume_volatility']
        for feature in illiquidity_features:
            if feature in df.columns:
                # Replace extreme values using existing robust scaling
                df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
                df[feature] = df[feature].fillna(df[feature].median())
                
                # Enhanced clipping using existing IQR method
                q1 = df[feature].quantile(0.01)
                q3 = df[feature].quantile(0.99)
                df[feature] = np.clip(df[feature], q1, q3)
                
                # Apply existing robust scaling
                median_val = df[feature].median()
                iqr = df[feature].quantile(0.75) - df[feature].quantile(0.25)
                if iqr > 0:
                    df[feature] = (df[feature] - median_val) / iqr
                
                print(f"🔄 Applied aggressive clipping to {feature}")
        
        return df

    def _apply_conservative_feature_selection(self, df):
        """Apply conservative feature selection to preserve important signals"""
        # Remove highly correlated features (less aggressive threshold)
        correlation_matrix = df.corr().abs()
        upper_tri = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Use 0.99 threshold instead of 0.95
        to_drop = [column for column in upper_tri.columns 
                  if any(upper_tri[column] > 0.99)]  # Less aggressive
        
        # 🟢 STRATEGIC FIX: Protect CRITICAL TIME-SERIES AND RISK INDICATORS
        protected_indicators = [
            'rsi_14', 'macd', 'macd_signal', 'bb_position', 
            'momentum_1m', 
            'volatility_5d', 'volatility_10d', 'volatility_20d', 'volatility_30d', # 🟢 RESTORED SHORT/LONG VOL
            'drawdown',                                                           # 🟢 RESTORED DRAWDOWN
            'stress_period'
        ]
        to_drop = [col for col in to_drop if col not in protected_indicators]
        
        if to_drop:
            print(f"🔄 Removing highly correlated features: {to_drop}")
            df = df.drop(columns=to_drop)
        
        # Conservative variance threshold - preserve regime features
        protected_features = [
            'stress_period', 'crisis_period', 'high_volatility', 
            'target_return_6m', 'target_direction_6m', 
            'target_return_1m', 'target_direction_1m',
            'drawdown', 'volatility_'                                              # 🟢 RESTORED FEATURES
        ]
        variance_threshold = 0.005  # More conservative
        
        low_variance_features = []
        for column in df.columns:
            if any(prot in column for prot in protected_features):
                continue
            if df[column].var() < variance_threshold:
                low_variance_features.append(column)
        
        # 🟢 FIX CHECK: Ensure critical features are NOT in low_variance_features
        low_variance_features = [
            f for f in low_variance_features if not any(
                p in f for p in ['drawdown', 'volatility_', 'price_velocity', 'price_acceleration']
            )
        ]
        
        if low_variance_features:
            print(f"🔄 Removing low variance features: {low_variance_features}")
            df = df.drop(columns=low_variance_features)
        
        return df

    def _add_robust_regime_features(self, df):
            """Enhanced regime feature engineering"""
            if 'returns' not in df.columns:
                df['returns'] = df['Close'].pct_change()
            
            returns = df['returns'].dropna()
            
            if len(returns) < 30:
                df['high_volatility'] = 0
                df['stress_period'] = 0
                return df
            
            # Multiple regime detection methods
            stress_methods = []
            
            # 1. Volatility-based stress
            vol_30d = returns.rolling(30).std().bfill().ffill()
            if len(vol_30d.dropna()) > 0:
                vol_threshold = vol_30d.quantile(0.80)  # More conservative threshold
                vol_stress = (vol_30d > vol_threshold).astype(int)
                stress_methods.append(vol_stress)
                df['high_volatility'] = vol_stress
            
            # 2. Drawdown-based stress
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            drawdown_stress = (drawdown < -0.08).astype(int)  # Less sensitive threshold
            stress_methods.append(drawdown_stress)
            df['drawdown'] = drawdown
            
            # 3. Price deviation stress
            price_sma_20 = df['Close'].rolling(20).mean()
            price_deviation = abs(df['Close'] - price_sma_20) / price_sma_20
            deviation_stress = (price_deviation > 0.08).astype(int)  # Conservative threshold
            stress_methods.append(deviation_stress)
            
            # Combine stress signals - require at least 2 out of 3
            if stress_methods:
                combined_stress = sum(stress_methods) >= 2
            else:
                combined_stress = pd.Series(False, index=df.index)
            
            # Ensure reasonable number of stress periods
            min_stress_required = max(20, int(len(df) * 0.03))  # More conservative
            max_stress_allowed = int(len(df) * 0.15)  # Cap stress periods
            
            current_stress = combined_stress.sum()
            
            if current_stress < min_stress_required:
                print(f"🔄 Adjusting stress periods from {current_stress} to {min_stress_required}")
                # Use volatility ranking to add stress periods
                vol_rank = vol_30d.rank(pct=True)
                additional_stress = (vol_rank > 0.90)  # Top 10% volatility
                combined_stress = combined_stress | additional_stress
            
            # Cap excessive stress periods
            if combined_stress.sum() > max_stress_allowed:
                print(f"🔄 Capping stress periods from {combined_stress.sum()} to {max_stress_allowed}")
                # Keep only the most extreme stress periods
                stress_scores = vol_30d.rank(pct=True) + drawdown.rank(pct=True)
                stress_threshold = stress_scores.quantile(1 - (max_stress_allowed / len(df)))
                combined_stress = (stress_scores > stress_threshold)
            
            df['stress_period'] = combined_stress.astype(int)
            final_stress_count = df['stress_period'].sum()
            stress_ratio = final_stress_count / len(df)
            
            print(f"📊 Regime detection: {final_stress_count} stress periods out of {len(df)} total ({stress_ratio:.1%})")
            
            return df



    def _fix_amihud_critical(self, df):
        """
        Final aggressive fix for amihud_illiquidity:
        1. Handle large outliers using sign-preserving log transformation.
        2. Aggressively clip the log-transformed data to a strict range.
        3. Standardize.
        """
        if 'amihud_illiquidity' not in df.columns:
            return df

        amihud = df['amihud_illiquidity'].copy()
        
        # --- FIX STEP 1: Handling extreme positive and negative values ---
        # Replace inf/nan with median (robust)
        amihud = amihud.replace([np.inf, -np.inf], np.nan).fillna(amihud.median())

        # Sign-preserving log transform (to compress extreme values)
        # log1p(abs(x)) * sign(x)
        amihud_log_compressed = np.log1p(np.abs(amihud)) * np.sign(amihud)
        
        # --- FIX STEP 2: Aggressive Clipping ---
        # Clip to a strict, small range (e.g., +/- 5 standard deviations from the median of the log data)
        # Use simple clipping to force values into a usable scale for the models
        amihud_fixed = np.clip(amihud_log_compressed, -5.0, 5.0) 
        
        # --- FIX STEP 3: Final Robust Scaling (for consistency) ---
        median_val = np.median(amihud_fixed)
        std_dev = np.std(amihud_fixed)
        
        # Standardize for mean ~ 0, std ~ 1
        if std_dev > 1e-8:
            amihud_fixed = (amihud_fixed - median_val) / std_dev
            
        df['amihud_illiquidity'] = amihud_fixed
        print("✅ CRITICAL FIX APPLIED: amihud_illiquidity successfully compressed and scaled.")
        return df

    def _remove_constant_features(self, df):
        constant_features = []
        protected_features = ['stress_period', 'high_volatility', 'crisis_period', 
                            'significant_drawdown', 'drawdown',
                            'target_direction_6m', 'target_direction_1m', 
                            'target_return_6m', 'target_return_1m']
        
        for col in df.columns:
            if col in protected_features:
                continue
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                if df[col].std() < 1e-8:
                    constant_features.append(col)
                elif df[col].nunique() <= 2:
                    unique_ratio = df[col].nunique() / len(df)
                    value_counts = df[col].value_counts(normalize=True)
                    if any(ratio > 0.99 for ratio in value_counts):
                        constant_features.append(col)
        
        if constant_features:
            print(f"🔄 Removing non-informative features: {constant_features}")
            df = df.drop(columns=constant_features)
        
        return df

    def _download_single_ticker(self, ticker, ticker_key):
        try:
            print(f"📥 Downloading {ticker_key}: {ticker}")
            data = yf.download(
                ticker, 
                start=self.config.START_DATE,
                end=self.config.END_DATE,
                progress=False
            )
            
            if data.empty:
                print(f"⚠️  No data for {ticker}")
                return False, None
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            
            required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
            available_cols = [col for col in required_cols if col in data.columns]
            
            if len(available_cols) < 3:
                print(f"⚠️  Insufficient columns for {ticker}")
                return False, None
            
            data = data[available_cols].dropna()
            
            if len(data) < 100:
                print(f"⚠️  Insufficient data points for {ticker}")
                return False, None
                
            print(f"✅ {ticker_key}: {len(data)} trading days")
            return True, data
            
        except Exception as e:
            print(f"❌ Failed to fetch {ticker}: {e}")
            return False, None

    def is_using_nzx50_index(self):
        if not self.nz50_ticker:
            return False
        return self.nz50_ticker.startswith('^') or 'NZ50' in self.nz50_ticker

    def get_market_coverage_info(self):
        nz_type = "NZX 50 Index" if self.is_using_nzx50_index() else "Individual NZ Stock"
        return {
            'nz_data_type': nz_type,
            'nz_ticker': self.nz50_ticker,
            'coverage': "Full market index" if self.is_using_nzx50_index() else "Single stock representation",
            'limitation': None if self.is_using_nzx50_index() else "Analysis based on single stock, not full market"
        }

    def analyze_market_crises(self, data):
        return self.crisis_detector.plot_crisis_timeline(data)


    def calculate_robust_sharpe_ratio(self, returns, risk_free_rate=0.02):
        if len(returns) < 2:
            return 0.0
        
        returns = np.array(returns)
        excess_returns = returns - (risk_free_rate / 252)
        std_dev = np.std(excess_returns)
        
        if std_dev < 1e-8:
            return 0.0
        
        sharpe_ratio = (np.mean(excess_returns) / std_dev) * np.sqrt(252)
        return float(sharpe_ratio)

    def calculate_max_drawdown(self, returns):
        if len(returns) < 2:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        return float(abs(max_drawdown))

    def analyze_usa_nz_correlations(self, market_data):
        print("🔍 Analyzing USA-NZ market correlations...")
        
        nzx_key = 'nz_primary' if 'nz_primary' in market_data else list(market_data.keys())[0]
        nzx_data = market_data[nzx_key]
        nzx_returns = nzx_data['Close'].pct_change().dropna()
        
        usa_tickers = {
            'S&P 500': 'usa_sp500',
            'NASDAQ': 'usa_nasdaq', 
            'Dow Jones': 'usa_dji'
        }
        
        correlation_results = {}
        for usa_name, usa_key in usa_tickers.items():
            if usa_key in market_data:
                usa_data = market_data[usa_key]
                usa_returns = usa_data['Close'].pct_change().dropna()
                
                aligned_returns = pd.concat([nzx_returns, usa_returns], axis=1, join='inner')
                aligned_returns.columns = ['NZX', 'USA']
                
                correlation = aligned_returns.corr().iloc[0, 1]
                correlation_results[usa_name] = correlation
                print(f"   • {usa_name}: {correlation:.3f}")
            else:
                print(f"⚠️  {usa_name} data not available")
        
        return correlation_results



    def enhanced_data_quality_check(self, forecast_data, feature_names):
        print("\n🔍 ENHANCED DATA QUALITY CHECK")
        print("=" * 50)
        
        sequences = forecast_data['sequences']
        targets_6m = forecast_data['targets_6m']
        
        print(f"📊 Sequences shape: {sequences.shape}")
        print(f"📊 Targets shape: {targets_6m.shape}")
        
        feature_means = sequences.mean(axis=(0, 1))
        feature_stds = sequences.std(axis=(0, 1))
        feature_mins = sequences.min(axis=(0, 1))
        feature_maxs = sequences.max(axis=(0, 1))
        
        problematic_features = []
        for i, (name, mean, std, min_val, max_val) in enumerate(zip(feature_names, feature_means, feature_stds, feature_mins, feature_maxs)):
            print(f"  {name:<25}: mean={mean:12.6f}, std={std:12.6f}, range=[{min_val:12.6f}, {max_val:12.6f}]")
            
            # Check for high magnitude features that need scaling
            if abs(mean) > 100 or std > 100 or abs(min_val) > 1000 or abs(max_val) > 1000:
                problematic_features.append((name, mean, std, min_val, max_val))
        
        target_returns = targets_6m[:, 0]
        print(f"\n🎯 TARGET ANALYSIS:")
        print(f"  Returns - Min: {target_returns.min():.6f}, Max: {target_returns.max():.6f}")
        print(f"  Returns - Mean: {target_returns.mean():.6f}, Std: {target_returns.std():.6f}")
        
        # 🆕 FIX: Check for NaN/Inf *after* initial feature engineering
        has_nan = np.isnan(sequences).any()
        has_inf = np.isinf(sequences).any()
        print(f"\n❓ DATA QUALITY:")
        print(f"  Contains NaN: {has_nan}, Contains Inf: {has_inf}")
        
        if has_nan or has_inf:
            print("🚨 CRITICAL: NaN/Inf detected! Normalization is required.")
            return False

        if problematic_features:
            print(f"\n🚨 PROBLEMATIC FEATURES (need robust normalization):")
            for name, mean, std, min_val, max_val in problematic_features:
                print(f"  {name}: mean={mean:.2f}, std={std:.2f}, range=[{min_val:.2f}, {max_val:.2f}]")
            return False
        
        constant_features = []
        for i, name in enumerate(feature_names):
            if sequences[:, :, i].std() < 1e-8:
                constant_features.append(name)
        
        if constant_features:
            print(f"\n⚠️  CONSTANT FEATURES (may be useless): {constant_features}")
        
        return True

    def normalize_features(self, forecast_data, feature_names):
        """Robust normalization using median and percentile for scale (more robust than mean/std)."""
        print("\n🔧 APPLYING FEATURE NORMALIZATION")
        
        sequences = forecast_data['sequences']
        normalized_sequences = sequences.copy()
        feature_means = []
        feature_stds = []
        
        for i in range(sequences.shape[2]):
            feature_data = sequences[:, :, i]
            
            # Replace NaN/Inf before normalization
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=1e5, neginf=-1e5)
            
            if feature_data.std() < 1e-8:
                feature_means.append(0)
                feature_stds.append(1)
                continue
            
            # Robust scaling: Use median for centering and 68th percentile of absolute deviation for scale
            mean = np.median(feature_data)
            std = np.percentile(np.abs(feature_data - mean), 68)
            
            if std < 1e-8:
                std = 1.0
                
            normalized_sequences[:, :, i] = (feature_data - mean) / std
            
            feature_means.append(mean)
            feature_stds.append(std)
            
            if i < 5:
                print(f"  {feature_names[i]}: median={mean:.6f} -> 0, std_approx={std:.6f} -> 1")
        
        # Final check for NaN/Inf after normalization
        if np.isnan(normalized_sequences).any() or np.isinf(normalized_sequences).any():
            print("🚨 CRITICAL: NaN/Inf still present after robust normalization!")
            # Fallback to standard min-max scaling for problematic features if needed
            
        forecast_data['sequences'] = normalized_sequences
        forecast_data['normalization'] = {
            'means': feature_means,
            'stds': feature_stds
        }
        
        print("✅ Feature normalization completed")
        return forecast_data
def create_forecast_loaders(forecast_data, batch_size=32):
    """Create data loaders for forecasting from processed data"""
    required_keys = ['sequences', 'targets_6m', 'targets_1m', 'regimes']
    for key in required_keys:
        if key not in forecast_data:
            raise ValueError(f"Missing required key in forecast_data: {key}")

    print(f"📊 Creating forecast loaders with {len(forecast_data['sequences'])} sequences")

    dataset = ForecastingDataset(
        forecast_data['sequences'],
        forecast_data['targets_6m'],
        forecast_data['targets_1m'],
        forecast_data['regimes']
    )

    split_labels = forecast_data.get('split_labels')
    if split_labels is None:
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        indices = list(range(total_size))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
    else:
        # Use explicit labels from time-series split (prevents leakage)
        train_indices = np.where(split_labels == 'train')[0]
        val_indices = np.where(split_labels == 'val')[0]
        test_indices = np.where(split_labels == 'test')[0]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    print(f"📊 Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    }

    feature_names = forecast_data.get('feature_names', [])

    print(f"✅ Forecast loaders created with {len(feature_names)} features")

    return loaders, feature_names