#!/usr/bin/env python3
"""
Adapter Trainer

Parameter-efficient fine-tuning system for neurosymbolic SQL adapters.
Handles training, validation, and optimization of LoRA adapters with
symbolic reasoning integration.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
import time
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import our neurosymbolic components
from .neurosymbolic_adapter import NeurosymbolicAdapter, AdapterConfig
from .bridge_layer import create_llama_bridge
from .confidence_estimator import ConfidenceEstimator, ConfidenceMethod
from .fact_extractor import create_llama_fact_extractor


@dataclass
class TrainingConfig:
    """Configuration for adapter training"""
    # Training parameters
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 100
    
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Neurosymbolic parameters
    bridge_dim: int = 256
    symbolic_dim: int = 512
    confidence_weight: float = 0.1  # Weight for confidence loss
    fact_consistency_weight: float = 0.05  # Weight for fact consistency loss
    
    # Optimization parameters
    use_gradient_checkpointing: bool = False
    use_mixed_precision: bool = False
    optimizer_type: str = "adamw"  # "adamw", "sgd", "adam"
    scheduler_type: str = "cosine"  # "cosine", "linear", "constant"
    
    # Validation parameters
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 50
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    
    # Output paths
    output_dir: str = "./output"
    save_total_limit: int = 3
    
    # Advanced features
    enable_symbolic_loss: bool = True
    enable_confidence_calibration: bool = True
    enable_fact_verification: bool = True


@dataclass
class TrainingState:
    """Current state of training"""
    epoch: int = 0
    global_step: int = 0
    best_loss: float = float('inf')
    best_metric: float = 0.0
    patience_counter: int = 0
    is_best_model: bool = False
    training_history: List[Dict[str, float]] = field(default_factory=list)
    validation_history: List[Dict[str, float]] = field(default_factory=list)


class SQLDataset(Dataset):
    """Dataset for SQL training examples"""
    
    def __init__(self, examples: List[Dict[str, Any]], tokenizer=None, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # For now, return mock tokenized data
        # In real implementation, this would use proper tokenizer
        input_text = example.get('instruction', '')
        target_sql = example.get('sql', '')
        schema = example.get('schema', '')
        
        # Mock tokenization
        input_ids = torch.randint(0, 1000, (min(len(input_text.split()), self.max_length),))
        labels = torch.randint(0, 1000, (min(len(target_sql.split()), self.max_length),))
        
        # Pad to max_length
        if len(input_ids) < self.max_length:
            padding = torch.zeros(self.max_length - len(input_ids), dtype=torch.long)
            input_ids = torch.cat([input_ids, padding])
        
        if len(labels) < self.max_length:
            padding = torch.full((self.max_length - len(labels),), -100, dtype=torch.long)
            labels = torch.cat([labels, padding])
        
        return {
            'input_ids': input_ids[:self.max_length],
            'labels': labels[:self.max_length],
            'attention_mask': torch.ones(self.max_length),
            'instruction': input_text,
            'sql': target_sql,
            'schema': schema
        }


class NeurosymbolicLoss(nn.Module):
    """Custom loss function for neurosymbolic training"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.base_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
    def forward(self, outputs, labels, confidence_scores=None, extracted_facts=None):
        """
        Compute comprehensive loss for neurosymbolic training
        
        Args:
            outputs: Model outputs with logits
            labels: Target labels
            confidence_scores: Confidence estimation results
            extracted_facts: Extracted symbolic facts
            
        Returns:
            Dictionary with loss components
        """
        losses = {}
        
        # Base language modeling loss
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        base_loss = self.base_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        losses['base_loss'] = base_loss
        
        # Confidence loss (encourage well-calibrated confidence)
        if confidence_scores is not None and self.config.confidence_weight > 0:
            confidence_loss = self._compute_confidence_loss(confidence_scores, shift_labels)
            losses['confidence_loss'] = confidence_loss * self.config.confidence_weight
        
        # Fact consistency loss
        if extracted_facts is not None and self.config.fact_consistency_weight > 0:
            fact_loss = self._compute_fact_consistency_loss(extracted_facts)
            losses['fact_consistency_loss'] = fact_loss * self.config.fact_consistency_weight
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_confidence_loss(self, confidence_scores, labels):
        """Compute loss for confidence calibration"""
        if hasattr(confidence_scores, 'overall_confidence'):
            # Encourage confidence to be inversely related to loss magnitude
            # High confidence should correlate with low prediction error
            confidence = confidence_scores.overall_confidence
            
            # Convert to tensor if it's a float
            if isinstance(confidence, float):
                confidence = torch.tensor(confidence)
            
            # Simple calibration loss
            target_confidence = torch.tensor(0.8)  # Target confidence for good predictions
            confidence_loss = torch.abs(confidence - target_confidence)
            
            return confidence_loss
        return torch.tensor(0.0)
    
    def _compute_fact_consistency_loss(self, extracted_facts):
        """Compute loss for fact consistency"""
        if hasattr(extracted_facts, 'extraction_confidence'):
            # Encourage high extraction confidence for consistent facts
            extraction_conf = extracted_facts.extraction_confidence
            fact_loss = torch.tensor(1.0 - extraction_conf)
            return fact_loss
        return torch.tensor(0.0)


class AdapterTrainer:
    """
    Comprehensive trainer for neurosymbolic SQL adapters
    
    Handles parameter-efficient fine-tuning with symbolic reasoning integration.
    """
    
    def __init__(self, config: TrainingConfig, model: Optional[NeurosymbolicAdapter] = None):
        """
        Initialize adapter trainer
        
        Args:
            config: Training configuration
            model: Pre-initialized model (optional)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize model if not provided
        if model is None:
            adapter_config = AdapterConfig(
                lora_r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bridge_dim=config.bridge_dim,
                symbolic_dim=config.symbolic_dim
            )
            self.model = NeurosymbolicAdapter(
                base_model_name="unsloth/llama-3.1-8b-instruct-bnb-4bit",
                config=adapter_config
            )
            
            # Initialize components
            self._initialize_neurosymbolic_components()
        else:
            self.model = model
        
        # Training state
        self.state = TrainingState()
        
        # Loss function
        self.loss_fn = NeurosymbolicLoss(config)
        
        # Optimizer and scheduler (will be initialized in setup_training)
        self.optimizer = None
        self.scheduler = None
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"AdapterTrainer initialized with config: {config}")
    
    def _initialize_neurosymbolic_components(self):
        """Initialize neurosymbolic components"""
        # Bridge layer
        bridge = create_llama_bridge()
        self.model.set_bridge_layer(bridge)
        
        # Confidence estimator
        confidence_estimator = ConfidenceEstimator(
            vocab_size=32000,
            hidden_dim=4096,
            symbolic_dim=self.config.symbolic_dim,
            methods=[ConfidenceMethod.ENTROPY, ConfidenceMethod.TEMPERATURE_SCALING]
        )
        self.model.set_confidence_estimator(confidence_estimator)
        
        # Fact extractor
        fact_extractor = create_llama_fact_extractor(
            extraction_threshold=0.5,
            max_facts_per_query=50
        )
        self.model.set_fact_extractor(fact_extractor)
        
        self.logger.info("Neurosymbolic components initialized")
    
    def setup_training(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Setup training components"""
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # Avoid multiprocessing issues
        )
        
        if eval_dataset:
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0
            )
        else:
            self.eval_loader = None
        
        # Optimizer
        if self.config.optimizer_type.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        
        # Scheduler
        total_steps = len(self.train_loader) * self.config.num_epochs
        
        if self.config.scheduler_type.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps
            )
        elif self.config.scheduler_type.lower() == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_steps
            )
        
        self.logger.info(f"Training setup complete: {len(self.train_loader)} batches per epoch")
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """
        Run complete training loop
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            
        Returns:
            Training results and metrics
        """
        self.setup_training(train_dataset, eval_dataset)
        
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        self.model.train()
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.state.epoch = epoch
            
            # Training epoch
            train_metrics = self._train_epoch()
            
            # Validation
            if self.eval_loader and (epoch + 1) % 1 == 0:  # Evaluate every epoch
                eval_metrics = self._evaluate()
                self.state.validation_history.append(eval_metrics)
                
                # Check for best model
                if eval_metrics['eval_loss'] < self.state.best_loss:
                    self.state.best_loss = eval_metrics['eval_loss']
                    self.state.is_best_model = True
                    self.state.patience_counter = 0
                    self._save_model(is_best=True)
                else:
                    self.state.patience_counter += 1
                    self.state.is_best_model = False
            
            # Save checkpoint
            if (epoch + 1) % (self.config.save_steps // len(self.train_loader)) == 0:
                self._save_model()
            
            # Early stopping
            if self.state.patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            self.logger.info(
                f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}, "
                f"lr={self.optimizer.param_groups[0]['lr']:.6f}"
            )
        
        training_time = time.time() - start_time
        
        # Final results
        results = {
            'training_time': training_time,
            'total_epochs': self.state.epoch + 1,
            'best_loss': self.state.best_loss,
            'training_history': self.state.training_history,
            'validation_history': self.state.validation_history,
            'final_model_path': str(self.output_dir / "final_model")
        }
        
        # Save final model
        self._save_model(final=True)
        
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        return results
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.state.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            self.state.global_step += 1
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # Compute loss
            loss_dict = self.loss_fn(
                outputs=outputs,
                labels=batch['labels'],
                confidence_scores=outputs.confidence_scores,
                extracted_facts=outputs.extracted_facts
            )
            
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Logging
            epoch_losses.append(total_loss.item())
            
            if self.state.global_step % self.config.logging_steps == 0:
                avg_loss = np.mean(epoch_losses[-self.config.logging_steps:])
                progress_bar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
        
        # Epoch metrics
        epoch_metrics = {
            'train_loss': np.mean(epoch_losses),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'epoch': self.state.epoch
        }
        
        self.state.training_history.append(epoch_metrics)
        return epoch_metrics
    
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set"""
        if not self.eval_loader:
            return {}
        
        self.model.eval()
        eval_losses = []
        confidence_scores = []
        fact_counts = []
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                loss_dict = self.loss_fn(
                    outputs=outputs,
                    labels=batch['labels'],
                    confidence_scores=outputs.confidence_scores,
                    extracted_facts=outputs.extracted_facts
                )
                
                eval_losses.append(loss_dict['total_loss'].item())
                
                # Collect additional metrics
                if outputs.confidence_scores:
                    confidence_scores.append(outputs.confidence_scores.overall_confidence)
                
                if outputs.extracted_facts:
                    fact_counts.append(len(outputs.extracted_facts.facts))
        
        eval_metrics = {
            'eval_loss': np.mean(eval_losses),
            'eval_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
            'eval_avg_facts': np.mean(fact_counts) if fact_counts else 0.0
        }
        
        return eval_metrics
    
    def _save_model(self, is_best: bool = False, final: bool = False):
        """Save model checkpoint"""
        if is_best:
            save_path = self.output_dir / "best_model"
        elif final:
            save_path = self.output_dir / "final_model"
        else:
            save_path = self.output_dir / f"checkpoint-{self.state.global_step}"
        
        # Save adapter
        self.model.save_adapter(save_path)
        
        # Save training state
        state_dict = {
            'epoch': self.state.epoch,
            'global_step': self.state.global_step,
            'best_loss': self.state.best_loss,
            'config': self.config.__dict__,
            'training_history': self.state.training_history,
            'validation_history': self.state.validation_history
        }
        
        with open(save_path / "training_state.json", 'w') as f:
            json.dump(state_dict, f, indent=2)
        
        self.logger.info(f"Model saved to {save_path}")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training and validation curves"""
        if not self.state.training_history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Training loss
        train_losses = [h['train_loss'] for h in self.state.training_history]
        axes[0, 0].plot(train_losses)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        
        # Validation loss
        if self.state.validation_history:
            val_losses = [h['eval_loss'] for h in self.state.validation_history]
            axes[0, 1].plot(val_losses)
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
        
        # Learning rate
        learning_rates = [h['learning_rate'] for h in self.state.training_history]
        axes[1, 0].plot(learning_rates)
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LR')
        
        # Confidence scores
        if self.state.validation_history:
            confidences = [h.get('eval_confidence', 0) for h in self.state.validation_history]
            axes[1, 1].plot(confidences)
            axes[1, 1].set_title('Validation Confidence')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Confidence')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig(self.output_dir / "training_curves.png")
        
        plt.close()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        summary = {
            'config': self.config.__dict__,
            'model_info': {
                'trainable_parameters': self.model.get_trainable_parameters(),
                'components': self.model.get_status()['components']
            },
            'training_state': {
                'current_epoch': self.state.epoch,
                'global_step': self.state.global_step,
                'best_loss': self.state.best_loss,
                'patience_counter': self.state.patience_counter
            },
            'performance': {
                'training_history': self.state.training_history[-5:],  # Last 5 epochs
                'validation_history': self.state.validation_history[-5:]  # Last 5 epochs
            }
        }
        
        return summary


# Convenience functions
def create_trainer(config: TrainingConfig, model: Optional[NeurosymbolicAdapter] = None) -> AdapterTrainer:
    """Create adapter trainer with configuration"""
    return AdapterTrainer(config, model)


def create_mock_dataset(num_examples: int = 100) -> SQLDataset:
    """Create mock dataset for testing"""
    examples = []
    
    for i in range(num_examples):
        examples.append({
            'instruction': f'Find all customers with more than {i % 10 + 1} orders',
            'sql': f'SELECT * FROM customers WHERE order_count > {i % 10 + 1}',
            'schema': 'customers (id, name, email, order_count)'
        })
    
    return SQLDataset(examples)


def train_neurosymbolic_adapter(config: TrainingConfig, train_data: List[Dict[str, Any]], 
                               eval_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """High-level training function"""
    # Create datasets
    train_dataset = SQLDataset(train_data)
    eval_dataset = SQLDataset(eval_data) if eval_data else None
    
    # Create trainer
    trainer = create_trainer(config)
    
    # Run training
    results = trainer.train(train_dataset, eval_dataset)
    
    # Generate plots
    trainer.plot_training_curves()
    
    return results