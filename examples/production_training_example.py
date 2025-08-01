#!/usr/bin/env python3
"""
Production Training Example for Neurosymbolic SQL Adapter

This example demonstrates production-ready training with:
- Robust error handling and recovery
- Monitoring and logging integration
- Checkpoint management
- Performance optimization
- Deployment preparation
"""

import sys
import yaml
import torch
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adapters.adapter_trainer import AdapterTrainer, TrainingConfig, create_mock_dataset
from adapters.neurosymbolic_adapter import NeurosymbolicAdapter, AdapterConfig
from adapters.model_manager import ModelManager, ModelConfig, ModelType, DeviceType
from adapters.confidence_estimator import ConfidenceMethod

class ProductionTrainingManager:
    """
    Production-ready training manager with comprehensive monitoring,
    error handling, and deployment preparation.
    """
    
    def __init__(self, config_path: str, run_id: Optional[str] = None):
        """
        Initialize production training manager
        
        Args:
            config_path: Path to training configuration
            run_id: Unique identifier for this training run
        """
        self.config_path = Path(config_path)
        self.run_id = run_id or f"prod_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Load configuration
        self.config = self._load_and_validate_config()
        
        # Setup directories
        self.output_dir = Path(f"./production_output/{self.run_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_production_logging()
        
        # Initialize training state
        self.training_state = {
            'status': 'initialized',
            'start_time': None,
            'current_epoch': 0,
            'best_metrics': {},
            'checkpoints': [],
            'errors': []
        }
        
        self.logger.info(f"Production training manager initialized: {self.run_id}")
    
    def _load_and_validate_config(self) -> Dict[str, Any]:
        """Load and validate training configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['model', 'lora', 'training', 'evaluation']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        return config
    
    def setup_production_logging(self):
        """Setup comprehensive production logging"""
        # Create logger
        self.logger = logging.getLogger(f"production_training_{self.run_id}")
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler
        log_file = self.output_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Production logging initialized: {log_file}")
    
    def save_training_state(self):
        """Save current training state to disk"""
        state_file = self.output_dir / "training_state.json"
        
        # Update timestamp
        self.training_state['last_updated'] = datetime.now().isoformat()
        
        with open(state_file, 'w') as f:
            json.dump(self.training_state, f, indent=2, default=str)
    
    def load_training_state(self) -> bool:
        """Load previous training state if exists"""
        state_file = self.output_dir / "training_state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    self.training_state = json.load(f)
                self.logger.info("Loaded previous training state")
                return True
            except Exception as e:
                self.logger.warning(f"Could not load training state: {e}")
        
        return False
    
    def create_production_model_config(self) -> ModelConfig:
        """Create production-optimized model configuration"""
        model_cfg = self.config['model']
        lora_cfg = self.config['lora']
        neurosymbolic_cfg = self.config.get('neurosymbolic', {})
        
        return ModelConfig(
            model_type=ModelType.LLAMA_8B,
            model_name=model_cfg['base_model'],
            device=DeviceType.AUTO,
            torch_dtype=model_cfg.get('torch_dtype', 'bfloat16'),
            
            # LoRA configuration
            lora_r=lora_cfg['r'],
            lora_alpha=lora_cfg['alpha'],
            lora_dropout=lora_cfg['dropout'],
            target_modules=lora_cfg['target_modules'],
            
            # Neurosymbolic configuration
            bridge_dim=neurosymbolic_cfg.get('bridge', {}).get('bridge_dim', 128),
            symbolic_dim=neurosymbolic_cfg.get('bridge', {}).get('symbolic_dim', 256),
            enable_bridge=True,
            enable_confidence=True,
            enable_fact_extraction=True,
            
            # Production optimizations
            load_in_4bit=model_cfg.get('quantization', {}).get('load_in_4bit', True),
            gradient_checkpointing=True
        )
    
    def create_production_training_config(self) -> TrainingConfig:
        """Create production-optimized training configuration"""
        train_cfg = self.config['training']
        
        return TrainingConfig(
            num_epochs=train_cfg['num_epochs'],
            batch_size=train_cfg['batch_size'],
            learning_rate=train_cfg['learning_rate'],
            weight_decay=train_cfg.get('weight_decay', 0.01),
            
            # Production settings
            gradient_clip_norm=train_cfg.get('gradient_clipping', {}).get('max_norm', 0.5),
            use_mixed_precision=train_cfg.get('mixed_precision', {}).get('enabled', True),
            
            # Monitoring
            eval_steps=self.config['evaluation']['eval_steps'],
            logging_steps=self.config['logging']['logging_steps'],
            save_steps=self.config['checkpointing']['save_steps'],
            
            # Output
            output_dir=str(self.output_dir / "model_checkpoints"),
            
            # Production-specific
            early_stopping_patience=5,
            save_total_limit=3
        )
    
    def setup_monitoring(self):
        """Setup production monitoring and alerting"""
        self.metrics_history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'memory_usage': [],
            'training_speed': []
        }
        
        # Create metrics file
        self.metrics_file = self.output_dir / "metrics.jsonl"
        
        self.logger.info("Production monitoring initialized")
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to file and update history"""
        # Add timestamp and step
        metrics_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            **metrics
        }
        
        # Append to metrics file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics_entry) + '\n')
        
        # Update history
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check system health and resource usage"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'memory_allocated': 0,
            'memory_reserved': 0,
            'disk_space_mb': 0
        }
        
        if torch.cuda.is_available():
            health_status['memory_allocated'] = torch.cuda.memory_allocated() / 1024**2
            health_status['memory_reserved'] = torch.cuda.memory_reserved() / 1024**2
        
        # Check disk space
        import shutil
        disk_usage = shutil.disk_usage(self.output_dir)
        health_status['disk_space_mb'] = disk_usage.free / 1024**2
        
        return health_status
    
    def create_production_datasets(self):
        """Create production training and evaluation datasets"""
        self.logger.info("Creating production datasets")
        
        # For this example, we'll create mock datasets
        # In production, these would be loaded from actual data files
        
        train_size = 1000  # Production would be much larger
        eval_size = 200
        
        train_dataset = create_mock_dataset(train_size)
        eval_dataset = create_mock_dataset(eval_size)
        
        self.logger.info(f"Created datasets: train={train_size}, eval={eval_size}")
        
        return train_dataset, eval_dataset
    
    def run_production_training(self) -> Dict[str, Any]:
        """Run complete production training pipeline"""
        try:
            self.training_state['status'] = 'running'
            self.training_state['start_time'] = datetime.now().isoformat()
            self.save_training_state()
            
            self.logger.info("Starting production training pipeline")
            
            # System health check
            health = self.check_system_health()
            self.logger.info(f"System health: GPU={health['gpu_available']}, Memory={health['memory_allocated']:.1f}MB")
            
            # Setup monitoring
            self.setup_monitoring()
            
            # Create configurations
            model_config = self.create_production_model_config()
            training_config = self.create_production_training_config()
            
            # Initialize model manager
            model_manager = ModelManager(model_config)
            
            # Load model
            model_id = f"production_model_{self.run_id}"
            adapter = model_manager.load_model(model_id, model_config)
            
            self.logger.info(f"Model loaded: {adapter.get_status()}")
            
            # Create trainer
            trainer = AdapterTrainer(training_config, adapter)
            
            # Create datasets
            train_dataset, eval_dataset = self.create_production_datasets()
            
            # Log initial metrics
            initial_metrics = {
                'train_dataset_size': len(train_dataset),
                'eval_dataset_size': len(eval_dataset),
                'model_parameters': adapter.get_trainable_parameters()
            }
            self.log_metrics(initial_metrics, 0)
            
            # Run training with monitoring
            self.logger.info("Starting training loop")
            training_start = time.time()
            
            results = trainer.train(train_dataset, eval_dataset)
            
            training_time = time.time() - training_start
            
            # Log final metrics
            final_metrics = {
                'total_training_time': training_time,
                'final_train_loss': results.get('training_history', [{}])[-1].get('train_loss', 0),
                'final_eval_loss': results.get('validation_history', [{}])[-1].get('eval_loss', 0),
                'total_epochs': results.get('total_epochs', 0)
            }
            self.log_metrics(final_metrics, results.get('total_epochs', 0))
            
            # Update training state
            self.training_state['status'] = 'completed'
            self.training_state['end_time'] = datetime.now().isoformat()
            self.training_state['final_metrics'] = final_metrics
            self.save_training_state()
            
            # Save final model
            final_model_path = self.output_dir / "final_model"
            adapter.save_adapter(final_model_path)
            
            # Create deployment package
            deployment_info = self.create_deployment_package(adapter, results)
            
            self.logger.info("Production training completed successfully")
            
            return {
                'status': 'success',
                'training_time': training_time,
                'model_path': str(final_model_path),
                'deployment_info': deployment_info,
                'metrics': final_metrics,
                'run_id': self.run_id
            }
            
        except Exception as e:
            # Handle training failure
            self.logger.error(f"Production training failed: {e}")
            
            self.training_state['status'] = 'failed'
            self.training_state['error'] = str(e)
            self.training_state['error_time'] = datetime.now().isoformat()
            self.save_training_state()
            
            return {
                'status': 'failed',
                'error': str(e),
                'run_id': self.run_id
            }
    
    def create_deployment_package(self, adapter: NeurosymbolicAdapter, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create deployment package with model and metadata"""
        self.logger.info("Creating deployment package")
        
        deployment_dir = self.output_dir / "deployment"
        deployment_dir.mkdir(exist_ok=True)
        
        # Model metadata
        model_metadata = {
            'model_id': f"neurosymbolic_sql_{self.run_id}",
            'version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'training_config': self.config,
            'model_config': adapter.get_status(),
            'training_results': {
                'total_epochs': training_results.get('total_epochs', 0),
                'best_loss': training_results.get('best_loss', float('inf')),
                'training_time': training_results.get('training_time', 0)
            },
            'deployment_requirements': {
                'python_version': '>=3.8',
                'torch_version': '>=2.0.0',
                'memory_gb': 8,
                'gpu_memory_gb': 4
            }
        }
        
        # Save metadata
        metadata_file = deployment_dir / "model_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(model_metadata, f, indent=2, default=str)
        
        # Copy model files
        model_files = deployment_dir / "model"
        model_files.mkdir(exist_ok=True)
        
        # Save adapter
        adapter.save_adapter(model_files)
        
        # Create deployment script
        deployment_script = deployment_dir / "deploy.py"
        with open(deployment_script, 'w') as f:
            f.write(self._generate_deployment_script())
        
        # Create Docker configuration
        dockerfile = deployment_dir / "Dockerfile"
        with open(dockerfile, 'w') as f:
            f.write(self._generate_dockerfile())
        
        self.logger.info(f"Deployment package created: {deployment_dir}")
        
        return {
            'deployment_dir': str(deployment_dir),
            'model_metadata': model_metadata,
            'files_created': [
                'model_metadata.json',
                'model/',
                'deploy.py',
                'Dockerfile'
            ]
        }
    
    def _generate_deployment_script(self) -> str:
        """Generate deployment script"""
        return '''#!/usr/bin/env python3
"""
Deployment script for Neurosymbolic SQL Adapter
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from adapters.neurosymbolic_adapter import load_pretrained_adapter
from integration.hybrid_model import NeurosymbolicSQLModel

def load_production_model():
    """Load production model for deployment"""
    model_path = Path(__file__).parent / "model"
    
    # Load adapter
    adapter = load_pretrained_adapter(model_path)
    
    # Create hybrid model
    model = NeurosymbolicSQLModel(
        enable_neural_adapters=True,
        base_model=adapter
    )
    
    return model

def generate_sql(instruction: str, schema: str = None):
    """Generate SQL using production model"""
    model = load_production_model()
    result = model.generate_sql(instruction, schema)
    
    return {
        'sql': result.sql,
        'confidence': result.confidence,
        'is_valid': result.is_valid,
        'explanation': result.explanation
    }

if __name__ == "__main__":
    # Example usage
    result = generate_sql("Find all customers with orders")
    print(f"Generated SQL: {result['sql']}")
    print(f"Confidence: {result['confidence']:.3f}")
'''
    
    def _generate_dockerfile(self) -> str:
        """Generate Dockerfile for deployment"""
        return '''FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and source code
COPY model/ ./model/
COPY src/ ./src/
COPY deploy.py .

# Expose port
EXPOSE 8000

# Run deployment
CMD ["python", "deploy.py"]
'''

def main():
    """Run production training example"""
    print("🏭 Production Training Example for Neurosymbolic SQL Adapter")
    print("=" * 70)
    
    # Use production configuration
    config_path = Path(__file__).parent.parent / "training_configs" / "production_config.yaml"
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return
    
    try:
        # Create production training manager
        training_manager = ProductionTrainingManager(str(config_path))
        
        # Run production training
        print(f"\n🚀 Starting production training run: {training_manager.run_id}")
        results = training_manager.run_production_training()
        
        # Display results
        if results['status'] == 'success':
            print(f"\n✅ Production training completed successfully!")
            print(f"   Run ID: {results['run_id']}")
            print(f"   Training time: {results['training_time']:.2f}s")
            print(f"   Model saved: {results['model_path']}")
            print(f"   Deployment ready: {results['deployment_info']['deployment_dir']}")
        else:
            print(f"\n❌ Production training failed:")
            print(f"   Error: {results['error']}")
            print(f"   Run ID: {results['run_id']}")
        
    except Exception as e:
        print(f"❌ Production training setup failed: {e}")
    
    print(f"\n🎉 Production training example completed!")

if __name__ == "__main__":
    main()