# model_architectures.py - ENHANCED PERFORMANCE VERSION

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import TrainingConfig
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin


class XGBoostForecaster(BaseEstimator, RegressorMixin):
    """XGBoost wrapper for the multi-horizon financial forecasting task."""
    def __init__(self, **kwargs):
        # 🆕 FIX: Remove explicit parameters that can be passed via kwargs
        # to avoid duplication
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 5,
            'random_state': 42
        }
        
        # 🆕 FIX: Update defaults with any provided kwargs
        default_params.update(kwargs)
        
        self.model = xgb.XGBRegressor(**default_params)

    def fit(self, X_flat, y_reg_6m):
        # X_flat should be (N, L*D), y_reg_6m should be (N,)
        self.model.fit(X_flat, y_reg_6m.ravel())
        return self

    def predict(self, X_flat):
        # Predict the 6-month regression return
        return self.model.predict(X_flat)
    
    # Mimic PyTorch model interface for compatibility with the main pipeline
    def __call__(self, X_seq, **kwargs):
        # 1. Convert PyTorch Tensor (B, L, D) to NumPy (B, L*D)
        X_seq_np = X_seq.cpu().numpy()
        X_flat = X_seq_np.reshape(X_seq_np.shape[0], -1)
        
        # 2. Get prediction (B,)
        reg_6m_pred_np = self.predict(X_flat)
        
        # 3. Convert back to PyTorch Tensor (B, 1) and create dummy outputs
        reg_6m = torch.from_numpy(reg_6m_pred_np).float().to(X_seq.device).unsqueeze(-1)
        
        # Dummy classification (Neutral 50/50 for the 6-month prediction)
        class_6m = torch.ones(reg_6m.shape[0], 2).to(X_seq.device) * 0.5
        
        # Mimic the MultiHorizonTransformer output structure (reg_6m, class_6m, reg_1m, class_1m, volatility, confidence)
        # For XGBoost, the 1-month is the same as 6-month
        return reg_6m, class_6m, reg_6m, class_6m, reg_6m * 0.1, reg_6m * 0.5
    
class AdvancedPositionalEncoding(nn.Module):
    """Enhanced positional encoding with learnable components"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(AdvancedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Fixed sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
        # Learnable positional adjustments
        self.learnable_pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        seq_len = x.size(1)
        fixed_pe = self.pe[:seq_len, :].transpose(0, 1)
        
        # Combine fixed and learnable positional encoding
        gate_weights = self.gate(x)
        enhanced_pe = fixed_pe + self.learnable_pe[:, :seq_len, :] * gate_weights
        
        x = x + enhanced_pe
        return self.dropout(x)

class EnhancedMultiHorizonTransformer(nn.Module):
    """Enhanced Transformer with advanced attention mechanisms and regime adaptation"""
    
    def __init__(self, feature_dim, d_model=128, nhead=8, num_layers=4, dropout=0.2):
        super(EnhancedMultiHorizonTransformer, self).__init__()
        
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.nhead = nhead
        
        # Enhanced input projection with residual connections
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Advanced positional encoding
        self.pos_encoding = AdvancedPositionalEncoding(d_model, dropout)
        
        # Multi-scale transformer encoder
        self.transformer_layers = nn.ModuleList([
            self._create_encoder_layer(d_model, nhead, dropout) 
            for _ in range(num_layers)
        ])
        
        # Temporal attention for sequence modeling
        self.temporal_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Regime adaptation layer
        self.regime_adapter = nn.Sequential(
            nn.Linear(d_model + 1, d_model),  # +1 for regime indicator
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Enhanced multi-horizon output heads
        self.forecast_6m_regression = self._create_output_head(d_model, 1)
        self.forecast_1m_regression = self._create_output_head(d_model, 1) 
        self.forecast_6m_classification = self._create_output_head(d_model, 2)
        self.forecast_1m_regression = self._create_output_head(d_model, 1)
        self.forecast_1m_classification = self._create_output_head(d_model, 2)
        
        # Advanced volatility forecasting
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Ensure positive volatility
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Attention weights storage
        self.attention_weights = None
        
        self.apply(self._init_weights)
    
    def _create_encoder_layer(self, d_model, nhead, dropout):
        return nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
    
    def _create_output_head(self, d_model, output_dim):
        return nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, output_dim)
        )
    
    def _init_weights(self, module):
        """Enhanced weight initialization"""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.constant_(module.in_proj_bias, 0)

    def forward(self, x, regime_indicator=None, return_attention=False):
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        
        # Incorporate regime information if available
        if regime_indicator is not None:
            # FIX: Ensure regime_indicator has shape [Batch, 1, 1] before expanding
            # Original: regime_expanded = regime_indicator.unsqueeze(-1).expand(-1, seq_len, -1)
            
            # This corrects the shape and performs the expansion correctly:
            regime_indicator = regime_indicator.unsqueeze(-1)  # [BatchSize, 1]
            regime_expanded = regime_indicator.unsqueeze(1).expand(-1, seq_len, -1) # [BatchSize, SeqLen, 1]
            
            x = torch.cat([x, regime_expanded], dim=-1)
            x = self.regime_adapter(x)
        
        # Apply positional encoding
        x = self.pos_encoding(x)
        
        # Process through transformer layers
        attention_weights_list = []
        
        if return_attention:
            # Capture attention weights with hooks
            def attention_hook(module, input, output):
                try:
                    if len(output) >= 2 and output[1] is not None:
                        attn_weights = output[1].detach()
                        if len(attn_weights.shape) == 3:
                            attn_weights = attn_weights.unsqueeze(1)
                        attention_weights_list.append(attn_weights)
                    else:
                        # Create proper dummy attention
                        dummy_attn = torch.eye(seq_len, device=x.device)
                        dummy_attn = dummy_attn.unsqueeze(0).unsqueeze(1)
                        dummy_attn = dummy_attn.repeat(batch_size, self.nhead, 1, 1)
                        attention_weights_list.append(dummy_attn)
                except Exception as e:
                    print(f"⚠️ Attention hook error: {e}")
                    dummy_attn = torch.eye(seq_len, device=x.device)
                    dummy_attn = dummy_attn.unsqueeze(0).unsqueeze(1)
                    dummy_attn = dummy_attn.repeat(batch_size, self.nhead, 1, 1)
                    attention_weights_list.append(dummy_attn)
            
            handles = []
            for layer in self.transformer_layers:
                handle = layer.self_attn.register_forward_hook(attention_hook)
                handles.append(handle)
            
            # Forward pass through all layers
            for layer in self.transformer_layers:
                x = layer(x)
            
            # Remove hooks
            for handle in handles:
                handle.remove()
                
            self.attention_weights = torch.stack(attention_weights_list) if attention_weights_list else None
        else:
            # Standard forward without attention capture
            for layer in self.transformer_layers:
                x = layer(x)
            self.attention_weights = None
        
        # Apply temporal attention
        temporal_out, temporal_attention = self.temporal_attention(
            x, x, x, need_weights=return_attention
        )
        
        if return_attention and temporal_attention is not None:
            if self.attention_weights is None:
                self.attention_weights = temporal_attention.unsqueeze(0)
            else:
                self.attention_weights = torch.cat([
                    self.attention_weights, 
                    temporal_attention.unsqueeze(0)
                ])
        
        # Enhanced pooling
        # ⚠️ CRITICAL: Unpack the two new outputs
        pooled_output_long, pooled_output_short = self._get_enhanced_pooled_output(x, temporal_out)

        # Multi-horizon predictions

        # 🆕 FIX: Use long-term focus for 6m horizon
        reg_6m = self.forecast_6m_regression(pooled_output_long)
        class_6m = self.forecast_6m_classification(pooled_output_long)

        # 🆕 FIX: Use short-term focus for 1m horizon
        reg_1m = self.forecast_1m_regression(pooled_output_short)
        class_1m = self.forecast_1m_classification(pooled_output_short)

        # Use the general long-term pooled output for volatility and confidence
        volatility = self.volatility_head(pooled_output_long)
        confidence = self.confidence_head(pooled_output_long)
        
        # Add calibrated noise during training for regularization
        if self.training:
            noise_scale = 0.01 * confidence
            reg_6m = reg_6m + torch.randn_like(reg_6m) * noise_scale
            reg_1m = reg_1m + torch.randn_like(reg_1m) * noise_scale
        
        if return_attention:
            return reg_6m, class_6m, reg_1m, class_1m, volatility, confidence, self.attention_weights
        else:
            return reg_6m, class_6m, reg_1m, class_1m, volatility, confidence

    # model_architectures.py - in EnhancedMultiHorizonTransformer (Method Refactor)

    def _get_enhanced_pooled_output(self, transformer_out, temporal_out):
        """
        Advanced pooling with multiple strategies.
        
        Refactored to return two distinct pooled outputs:
        1. pooled_output_long: Weighted towards global/long-term features (for 6m)
        2. pooled_output_short: Weighted towards recent/short-term features (for 1m)
        """
        batch_size, seq_len, d_model = transformer_out.shape
        
        # Strategy 1: Attention-based pooling (General/Global Context)
        try:
            temporal_attention_weights = F.softmax(
                temporal_out.mean(dim=-1), dim=-1
            ).unsqueeze(-1)
            attention_pooled = (transformer_out * temporal_attention_weights).sum(dim=1)
        except:
            attention_pooled = transformer_out.mean(dim=1)
        
        # Strategy 2: Multi-scale pooling
        # Last step (Short Term)
        last_step_pooled = transformer_out[:, -1, :]           
        # Global average (Long Term)
        global_average_pooled = transformer_out.mean(dim=1) 
        # Recent average (Medium Term)
        recent_average_pooled = transformer_out[:, -5:, :].mean(dim=1) # Use last 5 steps

        
        # 🆕 NEW FIX: Create two distinct vectors
        
        # Long-term focus (Heavier weight on global avg and attention)
        # Use global features + attention
        pooled_output_long = (
            0.5 * attention_pooled + 
            0.3 * global_average_pooled +
            0.2 * recent_average_pooled
        )
        
        # Short-term focus (Heavier weight on last step and recent avg)
        # Use recent features + attention
        pooled_output_short = (
            0.5 * last_step_pooled + 
            0.3 * recent_average_pooled + 
            0.2 * attention_pooled 
        )
        
        # ⚠️ CRITICAL: Return *both* long and short vectors
        return pooled_output_long, pooled_output_short

class AdaptiveLSTMModel(nn.Module):
    """Enhanced LSTM with attention and regime adaptation"""
    
    def __init__(self, feature_dim, hidden_size=128, num_layers=2, dropout=0.2):
        super(AdaptiveLSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            feature_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Regime adaptation
        self.regime_projection = nn.Linear(1, hidden_size * 2)
        
        # Output heads
        self.regression_head = self._create_output_head(hidden_size * 2, 1)
        self.classification_head = self._create_output_head(hidden_size * 2, 2)
        
        self.dropout = nn.Dropout(dropout)
    
    def _create_output_head(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x, regime_indicator=None):
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Incorporate regime information
        if regime_indicator is not None:
            regime_projected = self.regime_projection(regime_indicator.unsqueeze(-1))
            context_vector = context_vector + regime_projected.squeeze(1)
        
        # Apply dropout
        context_vector = self.dropout(context_vector)
        
        # Generate outputs
        regression_out = self.regression_head(context_vector)
        classification_out = self.classification_head(context_vector)
        
        return regression_out, classification_out

class EnhancedLinearBaseline(nn.Module):
    """Enhanced linear model with feature interactions"""
    
    def __init__(self, feature_dim, hidden_dim=64):
        super(EnhancedLinearBaseline, self).__init__()
        
        # Feature interaction layer
        self.feature_interaction = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Output heads
        self.regression_head = nn.Linear(hidden_dim // 2, 1)
        self.classification_head = nn.Linear(hidden_dim // 2, 2)
    
    def forward(self, x):
        # Use only the last time step
        last_features = x[:, -1, :]
        
        # Apply feature interactions
        enhanced_features = self.feature_interaction(last_features)
        
        # Generate outputs
        regression_out = self.regression_head(enhanced_features)
        classification_out = self.classification_head(enhanced_features)
        
        return regression_out, classification_out

class XGBoostForecaster(BaseEstimator, RegressorMixin):
    """XGBoost wrapper for the multi-horizon financial forecasting task."""
    def __init__(self, **kwargs):
        # 🆕 FIX: Remove explicit parameters that can be passed via kwargs
        # to avoid duplication
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 5,
            'random_state': 42
        }
        
        # 🆕 FIX: Update defaults with any provided kwargs
        default_params.update(kwargs)
        
        self.model = xgb.XGBRegressor(**default_params)
        
        # 🆕 ADD: PyTorch compatibility attributes
        self.device = torch.device('cpu')
        self.training = False

    def fit(self, X_flat, y_reg_6m):
        # X_flat should be (N, L*D), y_reg_6m should be (N,)
        self.model.fit(X_flat, y_reg_6m.ravel())
        return self

    def predict(self, X_flat):
        # Predict the 6-month regression return
        return self.model.predict(X_flat)
    
    # 🆕 ADD: PyTorch compatibility methods
    def parameters(self):
        """Return empty iterator to mimic PyTorch model interface"""
        return iter([])
    
    def named_parameters(self):
        """Return empty iterator to mimic PyTorch model interface"""
        return iter([])
    
    def to(self, device):
        """Mimic PyTorch .to() method for device placement"""
        self.device = device
        return self
    
    def train(self, mode=True):
        """Mimic PyTorch .train() method"""
        self.training = mode
        return self
    
    def eval(self):
        """Mimic PyTorch .eval() method"""
        self.training = False
        return self

    # Mimic PyTorch model interface for compatibility with the main pipeline
    def __call__(self, X_seq, **kwargs):
        # 1. Convert PyTorch Tensor (B, L, D) to NumPy (B, L*D)
        X_seq_np = X_seq.cpu().numpy()
        X_flat = X_seq_np.reshape(X_seq_np.shape[0], -1)
        
        # 2. Get prediction (B,)
        reg_6m_pred_np = self.predict(X_flat)
        
        # 3. Convert back to PyTorch Tensor (B, 1) and create dummy outputs
        reg_6m = torch.from_numpy(reg_6m_pred_np).float().to(X_seq.device).unsqueeze(-1)
        
        # Dummy classification (Neutral 50/50 for the 6-month prediction)
        class_6m = torch.ones(reg_6m.shape[0], 2).to(X_seq.device) * 0.5
        
        # Mimic the MultiHorizonTransformer output structure (reg_6m, class_6m, reg_1m, class_1m, volatility, confidence)
        # For XGBoost, the 1-month is the same as 6-month
        return reg_6m, class_6m, reg_6m, class_6m, reg_6m * 0.1, reg_6m * 0.5

class AdvancedModelFactory:
    """Enhanced model factory with performance optimizations"""
    
    @staticmethod
    def create_model(model_type, feature_dim, config=TrainingConfig(), **kwargs):
        """
        Create enhanced models based on type.
        
        Refactored to accept **kwargs and pass them to the XGBoostForecaster
        while ignoring them for other models.
        """
        if model_type == 'transformer':
            return EnhancedMultiHorizonTransformer(
                feature_dim=feature_dim,
                d_model=config.D_MODEL,
                nhead=config.NHEAD,
                num_layers=config.NUM_LAYERS,
                dropout=config.DROPOUT
            )
        elif model_type == 'lstm':
            return AdaptiveLSTMModel(
                feature_dim=feature_dim,
                hidden_size=config.D_MODEL,
                num_layers=2,
                dropout=config.DROPOUT
            )
        elif model_type == 'linear':
            return EnhancedLinearBaseline(
                feature_dim=feature_dim,
                hidden_dim=config.D_MODEL // 2
            )
        # 🆕 FIX: Handle XGBoost and pass extra **kwargs (n_estimators, max_depth, etc.)
        elif model_type == 'xgboost':
            # 🟢 FIXED: Remove feature_dim parameter since XGBoostForecaster doesn't use it
            return XGBoostForecaster(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_model_complexity(model):
        """Calculate model complexity with detailed breakdown"""
        # 🆕 FIX: Handle non-PyTorch models (like XGBoost) gracefully
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Detailed breakdown
            param_details = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param_details[name] = param.numel()
        else:
            # For non-PyTorch models like XGBoost, return 0 parameters
            total_params = 0
            param_details = {"note": "Non-PyTorch model (XGBoost), parameter count not available"}
        
        return total_params, param_details
    
    @staticmethod
    def create_ensemble(models_dict, ensemble_type='weighted'):
        """Create ensemble of models"""
        if ensemble_type == 'weighted':
            return WeightedModelEnsemble(models_dict)
        elif ensemble_type == 'stacking':
            return StackingEnsemble(models_dict)
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")

class WeightedModelEnsemble(nn.Module):
    """Weighted ensemble of models"""
    
    def __init__(self, models_dict):
        super(WeightedModelEnsemble, self).__init__()
        self.models = nn.ModuleDict(models_dict)
        
        # Learnable weights
        self.weights = nn.Parameter(
            torch.ones(len(models_dict)) / len(models_dict)
        )
    
    def forward(self, x, regime_indicator=None):
        predictions = []
        
        for name, model in self.models.items():
            if hasattr(model, 'forward_with_regime') and regime_indicator is not None:
                pred = model.forward_with_regime(x, regime_indicator)
            else:
                pred = model(x)
            predictions.append(pred)
        
        # Apply softmax to weights
        normalized_weights = F.softmax(self.weights, dim=0)
        
        # If predictions are tuples, combine each output separately
        if isinstance(predictions[0], tuple):
            num_outputs = len(predictions[0])
            ensemble_pred = tuple(
                sum(w * pred[i] for w, pred in zip(normalized_weights, predictions))
                for i in range(num_outputs)
            )
        else:
            ensemble_pred = sum(w * pred for w, pred in zip(normalized_weights, predictions))
        
        return ensemble_pred

class StackingEnsemble(nn.Module):
    """Stacking ensemble with meta-learner"""
    
    def __init__(self, models_dict, meta_input_size=None):
        super(StackingEnsemble, self).__init__()
        self.models = nn.ModuleDict(models_dict)
        
        if meta_input_size is None:
            meta_input_size = len(models_dict) * 2  # Regression + classification
        
        self.meta_learner = nn.Sequential(
            nn.Linear(meta_input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Regression + classification logits
        )
    
    def forward(self, x):
        # Get predictions from all models
        base_predictions = []
        for model in self.models.values():
            pred = model(x)
            if isinstance(pred, tuple):
                # Flatten regression and classification outputs
                base_predictions.extend([pred[0], pred[1]])
            else:
                # If output is not a tuple, assume regression only and create a dummy classification tensor
                dummy_class = torch.zeros_like(pred)
                base_predictions.extend([pred, dummy_class])
        
        # Stack predictions
        stacked_features = torch.cat(base_predictions, dim=-1)
        
        # Meta-learner prediction
        final_prediction = self.meta_learner(stacked_features)
        
        return final_prediction[:, 0:1], final_prediction[:, 1:]  # Regression, classification

# Backward compatibility
ModelFactory = AdvancedModelFactory
MultiHorizonTransformer = EnhancedMultiHorizonTransformer

