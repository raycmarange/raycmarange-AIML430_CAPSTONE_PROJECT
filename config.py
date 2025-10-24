# config.py - Enhanced Configuration with Symbolic Regression
# =============================================================================
# ROBUST CONFIGURATION WITH VERIFIED TICKER SYMBOLS
# =============================================================================
from datetime import date

class DataConfig:
    """Data configuration with verified ticker symbols"""
    # Comprehensive NZX 50 ticker options (in order of preference)
    NZ50_OPTIONS = [
        # Primary index symbols
        '^NZ50',           # Yahoo Finance NZX 50

        
        # Major individual constituents as fallbacks
        'AIA.NZ',          # Auckland Airport
        'FPH.NZ',          # Fisher & Paykel Healthcare
        'MFT.NZ',          # Mainfreight
        'SPK.NZ',          # Spark NZ
    ]
    
    # VERIFIED WORKING TICKERS (tested with yfinance)
    TICKERS = {
        'nz_primary': 'AIA.NZ',      # Reliable NZ stock with good history
        'usa_sp500': 'SPY',          # SPDR S&P 500 ETF
        'usa_nasdaq': 'QQQ',         # NASDAQ-100 ETF  
        'usa_dji': 'DIA',            # Dow Jones ETF
        'volatility': 'VXX'          # Volatility ETN
    }
    
    # Date ranges - extended for better training
    START_DATE = '2005-01-01'
    END_DATE = date.today().strftime('%Y-%m-%d')

    # Feature engineering - simplified for stability
    SEQUENCE_LENGTH = 60
    CORRELATION_WINDOWS = [30, 60]
    VOLATILITY_WINDOWS = [10, 20, 30]
    
    # Technical indicators
    TECHNICAL_INDICATORS = ['rsi', 'macd', 'bollinger_bands']

    # =============================================================================
    # STRESS_PERIODS: Focused on major global/local downturns impacting the NZX 50
    # =============================================================================
    STRESS_PERIODS = {
        # 1. Global Crisis Periods (2000-2025)
        'dot_com_burst': ('2000-03-01', '2002-10-01'),        # Dot-com bubble burst and post-9/11 downturn
        'global_financial_crisis': ('2007-07-01', '2009-03-31'), # GFC (worst crisis since Great Depression)
        'european_debt_crisis': ('2010-04-01', '2012-07-01'), # Follow-on instability from the GFC
        'covid_pandemic_crash': ('2020-02-20', '2020-04-07'), # Sharpest recent crash period
        
        # 2. Post-Pandemic/Local Inflation Cycle
        'post_covid_inflation': ('2022-01-01', '2023-12-31'), # Period of high inflation, RBNZ rate hikes, and recessionary pressures
        
        # 3. Local/Domestic Stress (Keep the previous NZX-specific periods)
        'nz_housing_correction_2018': ('2018-01-01', '2018-12-31'), # Local market stress/correction
        
        # 4. Normal Period Baseline
        'normal_growth_baseline': ('2013-01-01', '2019-12-31') # Period of generally low volatility before COVID
    
    }

# Add to config.py - Live Price Configuration
class LivePriceConfig:
    """Configuration for live price fetching"""
    LIVE_PRICE_SOURCES = {
        'yahoo': {
            'base_url': 'https://query1.finance.yahoo.com/v8/finance/chart/',
            'params': {'range': '1d', 'interval': '1m'}
        },
        'alpha_vantage': {
            'api_key': 'YOUR_ALPHA_VANTAGE_API_KEY',  # Optional fallback
            'base_url': 'https://www.alphavantage.co/query'
        }
    }
    
    # Primary ticker for live price fetching
    LIVE_TICKER = '^NZ50'  # NZX 50 Index
    
    # Fallback tickers if primary fails
    FALLBACK_TICKERS = ['AIA.NZ', 'FPH.NZ', 'MFT.NZ']
    
    # Request settings
    TIMEOUT = 10
    RETRY_ATTEMPTS = 2

class TrainingConfig:
    """OPTIMIZED training configuration for stable performance"""
    
    # Model architecture - OPTIMIZED for transformer stability
    D_MODEL = 64           
    NHEAD = 4              
    NUM_LAYERS = 2         # Reduced from 3 for better stability
    DROPOUT = 0.5         # Increased regularization
    SEQUENCE_LENGTH = 60
    
    # Training parameters - OPTIMIZED
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-5  
    EPOCHS = 100           
    STRESS_WEIGHT = 2.0    
    
    # Enhanced optimization
    OPTIMIZER = "adamw"
    SCHEDULER = "cosine_warmup"
    GRAD_CLIP = 0.5        # Tighter gradient clipping
    WEIGHT_DECAY = 1e-4
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 30
    VALIDATION_SPLIT = 0.2

# 🆕 INTEGRATED MODEL CONFIGURATION FUNCTION
class ModelConfig:
    """Dynamic model configuration creation"""
    
    @staticmethod
    def create_forecasting_model_configs(feature_dim):
        """Create configuration for forecasting models with stability improvements"""
        return {
            # ... transformer_forecaster remains the same ...
            'transformer_forecaster': {
                'type': 'transformer', 
                'feature_dim': feature_dim,
                'description': 'Multi-horizon Transformer for NZX 50',
                'epochs': 100,
                'stress_weight': 1.5,
                'learning_rate': 0.0002,
                'grad_clip': 0.5,
                'd_model': 64,
                'nhead': 4,
                'num_layers': 2
            },
            'lstm_forecaster': {
                'type': 'lstm', 
                'feature_dim': feature_dim,
                'description': 'LSTM for comparison',
                'epochs': 80,
                'stress_weight': 1.5,
                'learning_rate': 0.0005,
                'grad_clip': 0.5,
                # 🆕 FIX: Reduced model capacity (64->32) and added specific dropout (0.6) to reduce overfitting
                'hidden_size': 32, 
                'dropout': 0.6 
            },
            'linear_forecaster': {
                        'type': 'linear', 
                        'feature_dim': feature_dim,
                        'description': 'Linear baseline (Aggressively Stabilized)',
                        'epochs': 50,
                        'stress_weight': 1.0,
                        # 🆕 CRITICAL STABILIZATION FIXES:
                        'learning_rate': 1e-6,  # 📉 Aggressively lowered (from 1e-5 or 1e-4)
                        'grad_clip': 0.05,       # 🛡️ Extremely tight clipping (from 0.1)
                        'weight_decay': 1e-2     # ⚖️ High weight decay for strong L2 regularization
                    },  
            'xgboost_forecaster': {
                'type': 'xgboost', 
                'description': 'XGBoost tree-based baseline',
                'epochs': 1,
                'learning_rate': 0.08, 
                'n_estimators': 150, 
                'max_depth': 4
            }
        }

class SymbolicConfig:
    """🆕 Configuration for Symbolic Regression & Enhanced XAI"""
    
    # Symbolic Regression Parameters
    POPULATION_SIZE = 1000
    GENERATIONS = 10
    STOPPING_CRITERIA = 0.01
    P_CROSSOVER = 0.7
    P_SUBTREE_MUTATION = 0.1
    P_HOIST_MUTATION = 0.05
    P_POINT_MUTATION = 0.1
    MAX_SAMPLES = 0.8
    PARSIMONY_COEFFICIENT = 0.01
    RANDOM_STATE = 42
    
    # Function set for symbolic expressions
    FUNCTION_SET = ('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv')
    
    # Enhanced XAI Settings
    SHAP_SAMPLE_SIZE = 100        # Reduced for performance
    COUNTERFACTUAL_FEATURES = ['rsi_14', 'macd', 'momentum_6m', 'volatility_20d']
    COUNTERFACTUAL_CHANGE = 0.1   # 10% change for counterfactuals
    
    # Complexity scoring
    MAX_EQUATION_COMPLEXITY = 20
    IDEAL_COMPLEXITY_RANGE = (5, 15)
    
    # Interpretability thresholds
    HIGH_INTERPRETABILITY = 0.8
    MEDIUM_INTERPRETABILITY = 0.6
    LOW_INTERPRETABILITY = 0.4

class ForecastConfig:
    """Enhanced forecasting with uncertainty"""
    FORECAST_PERIODS = 126  # 6 months
    CONFIDENCE_LEVELS = [0.8, 0.9, 0.95]
    ENABLE_UNCERTAINTY = True  # New: probabilistic forecasts
    
    # 🆕 Updated Ensemble forecasting with XGBoost
    ENSEMBLE_WEIGHTS = {
        'transformer_forecaster': 0.35,
        'lstm_forecaster': 0.35, 
        'linear_forecaster': 0.2,
        'xgboost_forecaster': 0.1
    }
    
    # Confidence calibration
    MIN_ENSEMBLE_CONFIDENCE = 0.5
    CONFIDENCE_CALIBRATION = {
        'agreement_weight': 0.6,
        'magnitude_weight': 0.3,
        'volatility_weight': 0.1
    }

class ExperimentConfig:
    """Optimized experiment tracking"""
    PROJECT_NAME = "AIML430_Stable_Forecasting"
    EXPERIMENT_NAME = "enhanced_symbolic_xai"
    LOG_DIR = "./logs/"
    MODEL_DIR = "./saved_models/"
    RESULTS_DIR = "./results/"
    XAI_DIR = "./xai_visualizations/"
    
    # 🆕 Enhanced directories for symbolic analysis
    SYMBOLIC_DIR = "./symbolic_results/"
    COMPARISON_DIR = "./model_comparisons/"
    
    # Enhanced metrics
    TRACK_METRICS = [
        'mse', 'mae', 'direction_accuracy', 'prediction_correlation', 
        'sharpe_ratio', 'transparency_score', 'equation_complexity'
    ]
    
    # Performance monitoring
    PERFORMANCE_THRESHOLDS = {
        'min_direction_accuracy': 0.55,
        'max_val_loss_increase': 0.1,
        'min_prediction_correlation': 0.1,
        'min_transparency_score': 0.7  # 🆕 XAI requirement
    }
    
    # 🆕 Research Questions Tracking
    RESEARCH_QUESTIONS = {
        'RQ1': 'Performance during Stress Periods',
        'RQ2': 'Ethical Implications & Model Interpretability', 
        'RQ3': 'Explainable AI Method Comparison',
        'RQ4': 'Market Coverage & Data Limitations',
        'RQ5': 'Regulatory Alignment',
        'RQ6': 'Symbolic vs Neural Representations'  # 🆕 NEW RQ
    }

# 🆕 Package Requirements Configuration
class RequirementsConfig:
    """Package requirements for enhanced functionality"""
    
    # Core packages (existing)
    CORE_PACKAGES = {
        'torch': '>=2.0.0',
        'pandas': '>=1.5.0',
        'numpy': '>=1.21.0',
        'yfinance': '>=0.2.0',
        'sklearn': '>=1.2.0',
        'matplotlib': '>=3.5.0',
        'seaborn': '>=0.12.0'
    }
    
    # 🆕 Enhanced XAI & Symbolic Regression Packages
    XAI_PACKAGES = {
        'shap': '>=0.42.0',           # Existing SHAP
        'gplearn': '==0.4.2',         # 🆕 NEW: Symbolic Regression
        'sympy': '>=1.10',            # 🆕 NEW: Symbolic mathematics
        'interpret': '>=0.4.0'        # Optional: Additional interpretability
    }
    
    # Data processing & utilities
    UTILITY_PACKAGES = {
        'scipy': '>=1.9.0',
        'tqdm': '>=4.64.0',
        'plotly': '>=5.10.0',         # Enhanced visualizations
        'ipywidgets': '>=8.0.0'       # Interactive widgets for XAI
    }
    
    @classmethod
    def generate_requirements_file(cls, filename='requirements.txt'):
        """Generate requirements.txt file with all dependencies"""
        requirements = []
        
        # Add core packages
        for package, version in cls.CORE_PACKAGES.items():
            requirements.append(f"{package}{version}")
        
        # Add XAI packages
        for package, version in cls.XAI_PACKAGES.items():
            requirements.append(f"{package}{version}")
            
        # Add utility packages
        for package, version in cls.UTILITY_PACKAGES.items():
            requirements.append(f"{package}{version}")
        
        # Write to file
        with open(filename, 'w') as f:
            f.write("# Enhanced NZX 50 Forecasting Requirements\n")
            f.write("# Generated automatically from config.py\n\n")
            f.write("\n".join(requirements))
        
        print(f"✅ Requirements file generated: {filename}")
        return requirements

    @classmethod
    def validate_installation(cls):
        """Validate that all required packages are installed."""
        all_packages = {**cls.CORE_PACKAGES, **cls.XAI_PACKAGES, **cls.UTILITY_PACKAGES}
        missing = [pkg for pkg in all_packages if not cls._is_package_available(pkg)]

        if missing:
            print(f"❌ Missing packages: {', '.join(missing)}")
            print("💡 Run: pip install -r requirements.txt")
            return False

        print("✅ All required packages are installed")
        return True

    @staticmethod
    def _is_package_available(package_name):
        try:
            __import__(package_name)
            return True
        except ImportError:
            return False