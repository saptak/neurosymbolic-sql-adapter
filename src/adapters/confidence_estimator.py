#!/usr/bin/env python3
"""
Confidence Estimator

Uncertainty quantification module for neurosymbolic SQL generation.
Estimates confidence scores for generated SQL queries and symbolic reasoning.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np
from enum import Enum


class ConfidenceMethod(Enum):
    """Available confidence estimation methods"""
    ENTROPY = "entropy"
    TEMPERATURE_SCALING = "temperature_scaling"
    MONTE_CARLO_DROPOUT = "monte_carlo_dropout"
    ENSEMBLE = "ensemble"
    ATTENTION_BASED = "attention_based"
    SYMBOLIC_CONSISTENCY = "symbolic_consistency"


@dataclass
class ConfidenceOutput:
    """Output from confidence estimation"""
    overall_confidence: float  # Overall confidence score [0, 1]
    token_confidences: Optional[torch.Tensor] = None  # Per-token confidence scores
    uncertainty_estimate: Optional[float] = None  # Epistemic uncertainty
    calibration_score: Optional[float] = None  # Calibration quality
    method_scores: Optional[Dict[str, float]] = None  # Scores from different methods
    explanation: Optional[str] = None  # Human-readable explanation


class EntropyConfidenceEstimator(nn.Module):
    """Confidence estimation based on prediction entropy"""
    
    def __init__(self, vocab_size: int = 32000, temperature: float = 1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.temperature = temperature
        
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence based on prediction entropy
        
        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            
        Returns:
            Confidence scores [batch_size, seq_len]
        """
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        
        # Compute probabilities
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Compute entropy
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        # Convert entropy to confidence (lower entropy = higher confidence)
        max_entropy = np.log(self.vocab_size)
        confidence = 1.0 - (entropy / max_entropy)
        
        return torch.clamp(confidence, 0.0, 1.0)


class TemperatureScalingEstimator(nn.Module):
    """Temperature scaling for calibrated confidence estimation"""
    
    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))
        
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits
        
        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            
        Returns:
            Confidence scores [batch_size, seq_len]
        """
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Confidence is maximum probability
        max_probs, _ = torch.max(probs, dim=-1)
        return max_probs


class MonteCarloDropoutEstimator(nn.Module):
    """Monte Carlo Dropout for uncertainty estimation"""
    
    def __init__(self, dropout_rate: float = 0.1, num_samples: int = 10):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_samples = num_samples
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, hidden_states: torch.Tensor, 
                prediction_head: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate uncertainty using Monte Carlo Dropout
        
        Args:
            hidden_states: Hidden representations [batch_size, seq_len, hidden_dim]
            prediction_head: Model prediction head
            
        Returns:
            Mean predictions and uncertainty estimates
        """
        # Ensure model is in training mode for dropout
        was_training = prediction_head.training
        prediction_head.train()
        
        predictions = []
        
        for _ in range(self.num_samples):
            # Apply dropout and get predictions
            dropped_hidden = self.dropout(hidden_states)
            logits = prediction_head(dropped_hidden)
            probs = F.softmax(logits, dim=-1)
            predictions.append(probs)
        
        # Restore original training mode
        prediction_head.train(was_training)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # [num_samples, batch_size, seq_len, vocab_size]
        
        # Compute mean and variance
        mean_probs = predictions.mean(dim=0)
        var_probs = predictions.var(dim=0)
        
        # Compute predictive entropy (aleatoric uncertainty)
        predictive_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        
        # Compute mutual information (epistemic uncertainty)
        individual_entropies = -torch.sum(predictions * torch.log(predictions + 1e-8), dim=-1)
        expected_entropy = individual_entropies.mean(dim=0)
        mutual_info = predictive_entropy - expected_entropy
        
        # Overall uncertainty
        total_uncertainty = predictive_entropy
        
        # Convert to confidence
        max_entropy = np.log(mean_probs.size(-1))
        confidence = 1.0 - (total_uncertainty / max_entropy)
        
        return torch.clamp(confidence, 0.0, 1.0), mutual_info


class AttentionBasedEstimator(nn.Module):
    """Confidence estimation based on attention patterns"""
    
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Attention analysis network
        self.attention_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, attention_weights: torch.Tensor, 
                hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Estimate confidence based on attention patterns
        
        Args:
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
            
        Returns:
            Confidence scores [batch_size, seq_len]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        if attention_weights is not None:
            # Analyze attention dispersion
            # High dispersion = low confidence, focused attention = high confidence
            attention_entropy = self._compute_attention_entropy(attention_weights)
            
            # Combine with hidden state analysis
            hidden_confidence = self.attention_analyzer(hidden_states).squeeze(-1)
            
            # Weighted combination
            attention_confidence = 1.0 - attention_entropy
            combined_confidence = 0.7 * hidden_confidence + 0.3 * attention_confidence
        else:
            # Fallback to hidden state analysis only
            combined_confidence = self.attention_analyzer(hidden_states).squeeze(-1)
        
        return combined_confidence
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention weights"""
        # Average over heads
        avg_attention = attention_weights.mean(dim=1)  # [batch_size, seq_len, seq_len]
        
        # Compute entropy for each query position
        attention_entropy = -torch.sum(
            avg_attention * torch.log(avg_attention + 1e-8), 
            dim=-1
        )  # [batch_size, seq_len]
        
        # Normalize entropy
        max_entropy = np.log(avg_attention.size(-1))
        normalized_entropy = attention_entropy / max_entropy
        
        return normalized_entropy


class SymbolicConsistencyEstimator(nn.Module):
    """Confidence estimation based on symbolic reasoning consistency"""
    
    def __init__(self, symbolic_dim: int = 512):
        super().__init__()
        self.symbolic_dim = symbolic_dim
        
        # Consistency analysis network
        self.consistency_analyzer = nn.Sequential(
            nn.Linear(symbolic_dim, symbolic_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(symbolic_dim // 2, symbolic_dim // 4),
            nn.ReLU(),
            nn.Linear(symbolic_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, symbolic_embeddings: torch.Tensor, 
                extracted_facts: Optional[List[List[str]]] = None) -> torch.Tensor:
        """
        Estimate confidence based on symbolic consistency
        
        Args:
            symbolic_embeddings: Symbolic representations [batch_size, seq_len, symbolic_dim]
            extracted_facts: Extracted symbolic facts (optional)
            
        Returns:
            Confidence scores [batch_size, seq_len]
        """
        # Analyze symbolic representation coherence
        consistency_scores = self.consistency_analyzer(symbolic_embeddings).squeeze(-1)
        
        # If facts are provided, analyze fact consistency
        if extracted_facts:
            fact_consistency = self._analyze_fact_consistency(extracted_facts)
            
            # Expand fact consistency to match sequence length
            batch_size, seq_len = symbolic_embeddings.shape[:2]
            fact_consistency_expanded = fact_consistency.unsqueeze(1).expand(-1, seq_len)
            
            # Combine symbolic and fact consistency
            combined_confidence = 0.6 * consistency_scores + 0.4 * fact_consistency_expanded
        else:
            combined_confidence = consistency_scores
        
        return combined_confidence
    
    def _analyze_fact_consistency(self, extracted_facts: List[List[str]]) -> torch.Tensor:
        """Analyze consistency of extracted facts"""
        batch_size = len(extracted_facts)
        fact_scores = []
        
        for batch_facts in extracted_facts:
            if not batch_facts:
                fact_scores.append(0.5)  # Neutral score for no facts
                continue
            
            # Simple heuristic: more facts with higher confidence = better
            confidences = []
            for fact in batch_facts:
                if 'confidence=' in fact:
                    try:
                        conf_str = fact.split('confidence=')[1].rstrip(')')
                        confidence = float(conf_str)
                        confidences.append(confidence)
                    except:
                        confidences.append(0.5)
                else:
                    confidences.append(0.5)
            
            # Average confidence of facts
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            fact_scores.append(avg_confidence)
        
        return torch.tensor(fact_scores, dtype=torch.float32)


class ConfidenceEstimator(nn.Module):
    """
    Comprehensive Confidence Estimator
    
    Combines multiple uncertainty quantification methods for robust
    confidence estimation in neurosymbolic SQL generation.
    """
    
    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 4096, 
                 symbolic_dim: int = 512, methods: Optional[List[ConfidenceMethod]] = None):
        """
        Initialize confidence estimator
        
        Args:
            vocab_size: Vocabulary size of the language model
            hidden_dim: Hidden dimension of the language model
            symbolic_dim: Dimension of symbolic representations
            methods: List of confidence estimation methods to use
        """
        super().__init__()
        
        self.logger = logging.getLogger(__name__)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.symbolic_dim = symbolic_dim
        
        # Default methods if none specified
        if methods is None:
            methods = [
                ConfidenceMethod.ENTROPY,
                ConfidenceMethod.TEMPERATURE_SCALING,
                ConfidenceMethod.ATTENTION_BASED,
                ConfidenceMethod.SYMBOLIC_CONSISTENCY
            ]
        
        self.methods = methods
        
        # Initialize estimators
        self.estimators = nn.ModuleDict()
        
        if ConfidenceMethod.ENTROPY in methods:
            self.estimators['entropy'] = EntropyConfidenceEstimator(vocab_size)
        
        if ConfidenceMethod.TEMPERATURE_SCALING in methods:
            self.estimators['temperature'] = TemperatureScalingEstimator()
        
        if ConfidenceMethod.MONTE_CARLO_DROPOUT in methods:
            self.estimators['mc_dropout'] = MonteCarloDropoutEstimator()
        
        if ConfidenceMethod.ATTENTION_BASED in methods:
            self.estimators['attention'] = AttentionBasedEstimator(symbolic_dim)
        
        if ConfidenceMethod.SYMBOLIC_CONSISTENCY in methods:
            self.estimators['symbolic'] = SymbolicConsistencyEstimator(symbolic_dim)
        
        # Confidence fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(len(self.estimators), len(self.estimators) * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(len(self.estimators) * 2, 1),
            nn.Sigmoid()
        )
        
        self.logger.info(f"ConfidenceEstimator initialized with methods: {[m.value for m in methods]}")
    
    def forward(self, logits: torch.Tensor, hidden_states: torch.Tensor,
                attention_weights: Optional[torch.Tensor] = None,
                symbolic_embeddings: Optional[torch.Tensor] = None,
                extracted_facts: Optional[List[List[str]]] = None) -> ConfidenceOutput:
        """
        Estimate confidence using multiple methods
        
        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
            attention_weights: Attention weights (optional)
            symbolic_embeddings: Symbolic representations (optional)
            extracted_facts: Extracted facts (optional)
            
        Returns:
            ConfidenceOutput with comprehensive confidence estimates
        """
        batch_size, seq_len = logits.shape[:2]
        method_scores = {}
        individual_confidences = []
        
        # Apply each confidence estimation method
        if 'entropy' in self.estimators:
            entropy_conf = self.estimators['entropy'](logits)
            method_scores['entropy'] = entropy_conf.mean().item()
            individual_confidences.append(entropy_conf.unsqueeze(-1))
        
        if 'temperature' in self.estimators:
            temp_conf = self.estimators['temperature'](logits)
            method_scores['temperature'] = temp_conf.mean().item()
            individual_confidences.append(temp_conf.unsqueeze(-1))
        
        if 'attention' in self.estimators and symbolic_embeddings is not None:
            attention_conf = self.estimators['attention'](attention_weights, symbolic_embeddings)
            method_scores['attention'] = attention_conf.mean().item()
            individual_confidences.append(attention_conf.unsqueeze(-1))
        
        if 'symbolic' in self.estimators and symbolic_embeddings is not None:
            symbolic_conf = self.estimators['symbolic'](symbolic_embeddings, extracted_facts)
            method_scores['symbolic'] = symbolic_conf.mean().item()
            individual_confidences.append(symbolic_conf.unsqueeze(-1))
        
        # Combine confidence estimates
        if individual_confidences:
            # Stack individual confidences
            combined_confidences = torch.cat(individual_confidences, dim=-1)  # [batch_size, seq_len, num_methods]
            
            # Apply fusion network
            fused_confidence = self.fusion_network(combined_confidences).squeeze(-1)
            
            # Overall confidence (average over sequence)
            overall_confidence = fused_confidence.mean().item()
            
            # Uncertainty estimate (variance of predictions)
            uncertainty_estimate = fused_confidence.var().item()
        else:
            # Fallback to simple average
            overall_confidence = 0.5
            uncertainty_estimate = 0.5
            fused_confidence = torch.full((batch_size, seq_len), 0.5)
        
        # Generate explanation
        explanation = self._generate_explanation(method_scores, overall_confidence, uncertainty_estimate)
        
        return ConfidenceOutput(
            overall_confidence=overall_confidence,
            token_confidences=fused_confidence,
            uncertainty_estimate=uncertainty_estimate,
            calibration_score=self._estimate_calibration(fused_confidence),
            method_scores=method_scores,
            explanation=explanation
        )
    
    def _estimate_calibration(self, confidences: torch.Tensor) -> float:
        """Estimate calibration quality of confidence scores"""
        # Simple calibration heuristic: well-calibrated models should have
        # confidence scores distributed across the range [0, 1]
        conf_std = confidences.std().item()
        conf_mean = confidences.mean().item()
        
        # Good calibration: reasonable spread around 0.5-0.8 range
        ideal_mean = 0.65
        ideal_std = 0.15
        
        mean_error = abs(conf_mean - ideal_mean)
        std_error = abs(conf_std - ideal_std)
        
        # Convert errors to calibration score
        calibration_score = max(0.0, 1.0 - (mean_error + std_error))
        
        return calibration_score
    
    def _generate_explanation(self, method_scores: Dict[str, float], 
                            overall_confidence: float, uncertainty_estimate: float) -> str:
        """Generate human-readable explanation of confidence estimation"""
        explanation_parts = []
        
        # Overall assessment
        if overall_confidence > 0.8:
            explanation_parts.append("High confidence prediction")
        elif overall_confidence > 0.6:
            explanation_parts.append("Moderate confidence prediction")
        else:
            explanation_parts.append("Low confidence prediction")
        
        # Method-specific insights
        method_insights = []
        for method, score in method_scores.items():
            if score > 0.8:
                method_insights.append(f"{method}: very confident ({score:.2f})")
            elif score > 0.6:
                method_insights.append(f"{method}: moderately confident ({score:.2f})")
            else:
                method_insights.append(f"{method}: uncertain ({score:.2f})")
        
        if method_insights:
            explanation_parts.append("Method breakdown: " + ", ".join(method_insights))
        
        # Uncertainty assessment
        if uncertainty_estimate > 0.3:
            explanation_parts.append("High uncertainty detected - consider additional validation")
        elif uncertainty_estimate > 0.1:
            explanation_parts.append("Moderate uncertainty - reasonable prediction quality")
        else:
            explanation_parts.append("Low uncertainty - consistent predictions")
        
        return ". ".join(explanation_parts) + "."
    
    def calibrate_temperature(self, validation_logits: torch.Tensor, 
                            validation_labels: torch.Tensor) -> None:
        """Calibrate temperature scaling using validation data"""
        if 'temperature' not in self.estimators:
            self.logger.warning("Temperature scaling not available for calibration")
            return
        
        temp_estimator = self.estimators['temperature']
        
        # Simple calibration using NLL loss
        optimizer = torch.optim.LBFGS([temp_estimator.temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = validation_logits / temp_estimator.temperature
            loss = F.cross_entropy(
                scaled_logits.view(-1, scaled_logits.size(-1)),
                validation_labels.view(-1),
                ignore_index=-100
            )
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        self.logger.info(f"Temperature calibrated to: {temp_estimator.temperature.item():.3f}")
    
    def get_method_weights(self) -> Dict[str, float]:
        """Get learned weights for different confidence methods"""
        if hasattr(self.fusion_network[0], 'weight'):
            weights = self.fusion_network[0].weight[0].detach().cpu().numpy()
            method_names = list(self.estimators.keys())
            
            # Normalize weights
            weights = np.abs(weights)
            weights = weights / weights.sum()
            
            return dict(zip(method_names, weights))
        else:
            return {name: 1.0 / len(self.estimators) for name in self.estimators.keys()}
    
    def analyze_uncertainty_sources(self, logits: torch.Tensor, hidden_states: torch.Tensor,
                                  **kwargs) -> Dict[str, Any]:
        """Analyze different sources of uncertainty"""
        confidence_output = self.forward(logits, hidden_states, **kwargs)
        
        analysis = {
            'overall_confidence': confidence_output.overall_confidence,
            'uncertainty_estimate': confidence_output.uncertainty_estimate,
            'method_contributions': confidence_output.method_scores,
            'calibration_quality': confidence_output.calibration_score,
            'confidence_distribution': {
                'mean': confidence_output.token_confidences.mean().item(),
                'std': confidence_output.token_confidences.std().item(),
                'min': confidence_output.token_confidences.min().item(),
                'max': confidence_output.token_confidences.max().item()
            },
            'method_weights': self.get_method_weights()
        }
        
        return analysis


# Convenience functions
def create_confidence_estimator(vocab_size: int = 32000, hidden_dim: int = 4096, 
                               symbolic_dim: int = 512, **kwargs) -> ConfidenceEstimator:
    """Create a confidence estimator with standard configuration"""
    return ConfidenceEstimator(vocab_size, hidden_dim, symbolic_dim, **kwargs)


def create_llama_confidence_estimator(**kwargs) -> ConfidenceEstimator:
    """Create confidence estimator optimized for Llama models"""
    return ConfidenceEstimator(
        vocab_size=32000,  # Llama vocab size
        hidden_dim=4096,   # Llama hidden size
        symbolic_dim=512,
        **kwargs
    )