# ensemble_model.py - NEW FILE for confidence-boosted ensemble
import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

class ConfidenceBoostedEnsemble:
    """Enhanced ensemble with confidence boosting techniques"""
    
    def __init__(self, models, confidence_threshold=0.6):
        self.models = models
        self.confidence_threshold = confidence_threshold
        self.regime_detector = self._build_regime_detector()
        self.ensemble_weights = self._initialize_weights()
    
    def _build_regime_detector(self):
        """Build market regime detector"""
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=50, random_state=42)
    
    def _initialize_weights(self):
        """Initialize ensemble weights based on model performance"""
        return {name: 1.0/len(self.models) for name in self.models.keys()}
    
    def fit_regime_detector(self, historical_data, market_regimes):
        """Fit regime detector on historical data"""
        try:
            # Flatten sequential data for regime detection
            if len(historical_data.shape) == 3:
                historical_flat = historical_data.reshape(historical_data.shape[0], -1)
            else:
                historical_flat = historical_data
            
            self.regime_detector.fit(historical_flat, market_regimes)
            print("✅ Regime detector trained successfully")
        except Exception as e:
            print(f"⚠️ Regime detector training failed: {e}")
    
    def predict_with_confidence(self, current_data, historical_data):
        """Generate predictions with boosted confidence"""
        try:
            # Get individual model predictions
            base_predictions = {}
            for name, model in self.models.items():
                model.eval()
                with torch.no_grad():
                    if hasattr(model, 'regime_adapter'):
                        # For models that support regime indicators
                        pred = model(current_data, regime_indicator=None)
                    else:
                        pred = model(current_data)
                    
                    # Extract regression prediction
                    if isinstance(pred, (list, tuple)):
                        base_predictions[name] = pred[0].cpu().numpy().flatten()[0]
                    else:
                        base_predictions[name] = pred.cpu().numpy().flatten()[0]
            
            # Bayesian ensemble averaging
            weighted_pred = self._bayesian_ensemble(base_predictions)
            
            # Regime-specific adjustment
            current_regime = self._detect_current_regime(current_data)
            regime_adjusted_pred = self._apply_regime_adjustment(weighted_pred, current_regime)
            
            # Confidence calculation
            confidence = self._calculate_prediction_confidence(base_predictions, current_regime)
            
            # Apply conservative adjustment if low confidence
            if confidence < self.confidence_threshold:
                print("⚠️ Low confidence - applying conservative adjustment")
                final_prediction = self._apply_conservative_adjustment(regime_adjusted_pred, base_predictions)
                confidence = max(confidence, 0.3)  # Minimum confidence floor
            else:
                final_prediction = regime_adjusted_pred
            
            return final_prediction, confidence, current_regime
            
        except Exception as e:
            print(f"❌ Confidence-boosted prediction failed: {e}")
            # Fallback to simple average
            fallback_pred = np.mean(list(base_predictions.values()))
            return fallback_pred, 0.3, "Unknown"
    
    def _bayesian_ensemble(self, predictions):
        """Bayesian model averaging"""
        weights = np.array(list(self.ensemble_weights.values()))
        preds = np.array(list(predictions.values()))
        
        # Simple weighted average
        weighted_pred = np.sum(weights * preds)
        return weighted_pred
    
    # In ensemble_model.py - Debug regime detection
    def _detect_current_regime(self, current_data):
        """Debug regime detection"""
        try:
            if len(current_data.shape) == 3:
                current_flat = current_data.cpu().numpy().reshape(1, -1)
            else:
                current_flat = current_data.cpu().numpy()
            
            print(f"🔍 Regime detection - Input shape: {current_flat.shape}")
            print(f"🔍 Regime detector trained: {hasattr(self.regime_detector, 'classes_')}")
            
            regime_pred = self.regime_detector.predict(current_flat)[0]
            regime_proba = self.regime_detector.predict_proba(current_flat)[0]
            
            print(f"🔍 Regime prediction: {regime_pred}, Probabilities: {regime_proba}")
            
            return "Stress" if regime_pred == 1 else "Normal"
        except Exception as e:
            print(f"❌ Regime detection failed: {e}")
            return "Unknown"
    
    def _apply_regime_adjustment(self, prediction, regime):
        """Apply regime-specific adjustments"""
        if regime == "Stress":
            # More conservative predictions in stress periods
            return prediction * 0.8
        elif regime == "Normal":
            return prediction
        else:
            return prediction * 0.9  # Slightly conservative for unknown regime
    
    def _calculate_prediction_confidence(self, predictions, regime):
        """Enhanced confidence calculation with more factors"""
        pred_values = list(predictions.values())
        
        # 1. Variance-based confidence
        variance = np.var(pred_values)
        variance_confidence = 1 - min(variance, 0.1)  # Cap variance impact
        
        # 2. Agreement-based confidence  
        mean_pred = np.mean(pred_values)
        agreements = sum(1 for p in pred_values if abs(p - mean_pred) < 0.005)  # Tighter agreement
        agreement_ratio = agreements / len(pred_values)
        
        # 3. Magnitude-based confidence (stronger signals more confident)
        signal_strength = min(abs(mean_pred) * 10, 1.0)  # Scale to 0-1
        
        # 4. Model performance weighting (use historical performance)
        performance_weights = self._get_model_performance_weights()
        performance_confidence = np.mean(list(performance_weights.values()))
        
        # Combined confidence with better weighting
        base_confidence = (
            variance_confidence * 0.3 +
            agreement_ratio * 0.3 + 
            signal_strength * 0.2 +
            performance_confidence * 0.2
        )
        
        # Regime adjustment
        regime_multiplier = 0.8 if regime == "Stress" else 1.0 if regime == "Normal" else 0.7
        
        final_confidence = base_confidence * regime_multiplier
        
        return min(final_confidence, 0.8)  # Cap at 80% for realism
    
    def _apply_conservative_adjustment(self, prediction, base_predictions):
        """Apply conservative adjustment for low-confidence predictions"""
        # Use median of predictions for robustness
        median_pred = np.median(list(base_predictions.values()))
        
        # Blend towards zero (no change) for conservatism
        conservative_pred = 0.3 * median_pred + 0.7 * 0.0
        
        return conservative_pred
    
    def update_weights_based_on_performance(self, performance_metrics):
        """Update ensemble weights based on recent performance"""
        try:
            total_performance = sum(performance_metrics.values())
            if total_performance > 0:
                for model_name in self.ensemble_weights:
                    if model_name in performance_metrics:
                        self.ensemble_weights[model_name] = (
                            performance_metrics[model_name] / total_performance
                        )
            print("✅ Ensemble weights updated based on performance")
        except Exception as e:
            print(f"⚠️ Weight update failed: {e}")