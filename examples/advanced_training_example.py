#!/usr/bin/env python3
"""
Advanced Training Example for Neurosymbolic SQL Adapter

This example demonstrates advanced training techniques including:
- Custom training configurations
- Multi-stage curriculum learning
- Advanced loss functions
- Comprehensive evaluation
- Production-ready training pipeline
"""

import sys
import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adapters.adapter_trainer import AdapterTrainer, TrainingConfig
from adapters.neurosymbolic_adapter import NeurosymbolicAdapter, AdapterConfig
from adapters.model_manager import ModelManager, ModelConfig, ModelType, DeviceType
from adapters.confidence_estimator import ConfidenceMethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedTrainingPipeline:
    """
    Advanced training pipeline with comprehensive configuration management
    and multi-stage training capabilities.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize advanced training pipeline
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.setup_logging()
        
        # Initialize components
        self.model_config = self._create_model_config()
        self.training_config = self._create_training_config()
        self.model_manager = None
        self.trainer = None
        
        logger.info(f"Advanced training pipeline initialized with config: {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {self.config_path}")
        return config
    
    def setup_logging(self):
        """Setup advanced logging configuration"""
        log_config = self.config.get('logging', {})
        
        # Configure log level
        log_level = getattr(logging, log_config.get('log_level', 'INFO'))
        logging.getLogger().setLevel(log_level)
        
        # Setup file logging if specified
        if 'log_file' in log_config:
            file_handler = logging.FileHandler(log_config['log_file'])
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logging.getLogger().addHandler(file_handler)
    
    def _create_model_config(self) -> ModelConfig:
        """Create model configuration from YAML config"""
        model_cfg = self.config['model']
        lora_cfg = self.config['lora']
        neurosymbolic_cfg = self.config['neurosymbolic']
        
        # Map confidence methods
        confidence_methods = []
        for method in neurosymbolic_cfg['confidence']['methods']:
            if method == "entropy":
                confidence_methods.append(ConfidenceMethod.ENTROPY)
            elif method == "temperature_scaling":
                confidence_methods.append(ConfidenceMethod.TEMPERATURE_SCALING)
            elif method == "attention_based":
                confidence_methods.append(ConfidenceMethod.ATTENTION_BASED)
        
        return ModelConfig(
            model_type=ModelType.LLAMA_8B if "llama-8b" in model_cfg['model_type'] else ModelType.LLAMA_7B,
            model_name=model_cfg['base_model'],
            device=DeviceType.AUTO if model_cfg['device'] == 'auto' else DeviceType.CPU,
            torch_dtype=model_cfg['torch_dtype'],
            
            # LoRA settings
            lora_r=lora_cfg['r'],
            lora_alpha=lora_cfg['alpha'],
            lora_dropout=lora_cfg['dropout'],
            target_modules=lora_cfg['target_modules'],
            
            # Neurosymbolic settings
            bridge_dim=neurosymbolic_cfg['bridge']['bridge_dim'],
            symbolic_dim=neurosymbolic_cfg['bridge']['symbolic_dim'],
            enable_bridge=True,
            enable_confidence=True,
            enable_fact_extraction=True,
            
            confidence_methods=confidence_methods,
            fact_extraction_threshold=neurosymbolic_cfg['fact_extraction']['extraction_threshold'],
            max_facts_per_query=neurosymbolic_cfg['fact_extraction']['max_facts_per_query'],
            
            # Quantization
            load_in_4bit=model_cfg['quantization']['load_in_4bit'],
            bnb_4bit_compute_dtype=model_cfg['quantization']['bnb_4bit_compute_dtype']
        )
    
    def _create_training_config(self) -> TrainingConfig:
        """Create training configuration from YAML config"""
        train_cfg = self.config['training']
        loss_cfg = self.config['loss']
        
        return TrainingConfig(
            num_epochs=train_cfg['num_epochs'],
            batch_size=train_cfg['batch_size'],
            learning_rate=train_cfg['learning_rate'],
            weight_decay=train_cfg['weight_decay'],
            
            # LoRA settings
            lora_r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['alpha'],
            lora_dropout=self.config['lora']['dropout'],
            
            # Neurosymbolic settings
            bridge_dim=self.config['neurosymbolic']['bridge']['bridge_dim'],
            symbolic_dim=self.config['neurosymbolic']['bridge']['symbolic_dim'],
            
            # Loss weights
            confidence_weight=loss_cfg['confidence_loss']['weight'],
            fact_consistency_weight=loss_cfg['fact_consistency_loss']['weight'],
            
            # Training settings
            gradient_clip_norm=train_cfg['gradient_clipping']['max_norm'],
            use_mixed_precision=train_cfg['mixed_precision']['enabled'],
            optimizer_type=train_cfg['optimizer']['type'],
            scheduler_type=train_cfg['scheduler']['type'],
            
            # Evaluation
            eval_steps=self.config['evaluation']['eval_steps'],
            logging_steps=self.config['logging']['logging_steps'],
            
            # Checkpointing
            save_steps=self.config['checkpointing']['save_steps'],
            output_dir=f"./training_output_{self.config_path.stem}"
        )
    
    def setup_model_manager(self):
        """Setup model manager with configuration"""
        self.model_manager = ModelManager(self.model_config)
        logger.info("Model manager initialized")
    
    def create_curriculum_datasets(self) -> List[Dict[str, Any]]:
        """
        Create curriculum learning datasets based on configuration
        
        Returns:
            List of dataset configurations for each curriculum stage
        """
        curriculum_cfg = self.config.get('advanced', {}).get('curriculum_learning', {})
        
        if not curriculum_cfg.get('enabled', False):
            # Return single dataset if curriculum learning disabled
            return [{
                'name': 'standard',
                'path': self.config['data']['train_dataset_path'],
                'epochs': self.training_config.num_epochs,
                'complexity': 'mixed'
            }]
        
        datasets = []
        for stage in curriculum_cfg['stages']:
            datasets.append({
                'name': stage['name'],
                'epochs': stage['epochs'],
                'complexity_level': stage['complexity_level'],
                'schema_complexity': stage.get('schema_complexity', 'mixed'),
                'data': self._generate_curriculum_data(stage)
            })
        
        logger.info(f"Created {len(datasets)} curriculum learning stages")
        return datasets
    
    def _generate_curriculum_data(self, stage: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate curriculum data for a specific stage"""
        complexity = stage['complexity_level']
        
        # Generate sample data based on complexity level
        if complexity == 1:
            # Simple SELECT queries
            return [
                {
                    'instruction': 'Find all customers',
                    'sql': 'SELECT * FROM customers',
                    'schema': 'customers (id, name, email)'
                },
                {
                    'instruction': 'Get customer names',
                    'sql': 'SELECT name FROM customers',
                    'schema': 'customers (id, name, email)'
                }
            ] * 50  # Repeat for sufficient training data
            
        elif complexity == 2:
            # JOIN queries
            return [
                {
                    'instruction': 'Find customers with orders',
                    'sql': 'SELECT c.name, o.order_date FROM customers c JOIN orders o ON c.id = o.customer_id',
                    'schema': 'customers (id, name, email), orders (id, customer_id, order_date, amount)'
                },
                {
                    'instruction': 'Get customer order amounts',
                    'sql': 'SELECT c.name, o.amount FROM customers c JOIN orders o ON c.id = o.customer_id',
                    'schema': 'customers (id, name, email), orders (id, customer_id, order_date, amount)'
                }
            ] * 50
            
        elif complexity == 3:
            # Aggregation queries
            return [
                {
                    'instruction': 'Count orders per customer',
                    'sql': 'SELECT c.name, COUNT(o.id) FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name',
                    'schema': 'customers (id, name, email), orders (id, customer_id, order_date, amount)'
                },
                {
                    'instruction': 'Total sales by customer',
                    'sql': 'SELECT c.name, SUM(o.amount) FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name',
                    'schema': 'customers (id, name, email), orders (id, customer_id, order_date, amount)'
                }
            ] * 50
            
        elif complexity >= 4:
            # Complex queries with subqueries
            return [
                {
                    'instruction': 'Find customers with above average order amounts',
                    'sql': 'SELECT c.name FROM customers c WHERE c.id IN (SELECT o.customer_id FROM orders o WHERE o.amount > (SELECT AVG(amount) FROM orders))',
                    'schema': 'customers (id, name, email), orders (id, customer_id, order_date, amount)'
                },
                {
                    'instruction': 'Get top customers by total spending',
                    'sql': 'SELECT c.name, (SELECT SUM(o.amount) FROM orders o WHERE o.customer_id = c.id) as total FROM customers c ORDER BY total DESC LIMIT 10',
                    'schema': 'customers (id, name, email), orders (id, customer_id, order_date, amount)'
                }
            ] * 50
        
        return []
    
    def run_advanced_training(self):
        """Run complete advanced training pipeline"""
        logger.info("Starting advanced training pipeline")
        
        # Setup model manager
        self.setup_model_manager()
        
        # Load model
        model_id = "neurosymbolic_sql_model"
        adapter = self.model_manager.load_model(model_id, self.model_config)
        
        # Create trainer
        self.trainer = AdapterTrainer(self.training_config, adapter)
        
        # Get curriculum datasets
        curriculum_datasets = self.create_curriculum_datasets()
        
        # Run curriculum training
        total_training_time = 0
        for stage_idx, stage_data in enumerate(curriculum_datasets):
            logger.info(f"Starting curriculum stage {stage_idx + 1}: {stage_data['name']}")
            
            # Create dataset for this stage
            from adapters.adapter_trainer import SQLDataset
            stage_dataset = SQLDataset(stage_data['data'])
            
            # Update training config for this stage
            stage_config = TrainingConfig(
                num_epochs=stage_data['epochs'],
                batch_size=self.training_config.batch_size,
                learning_rate=self.training_config.learning_rate,
                output_dir=f"{self.training_config.output_dir}/stage_{stage_idx + 1}"
            )
            
            # Create new trainer for this stage
            stage_trainer = AdapterTrainer(stage_config, adapter)
            
            # Run training for this stage
            stage_results = stage_trainer.train(stage_dataset)
            total_training_time += stage_results['training_time']
            
            logger.info(f"Completed stage {stage_idx + 1} in {stage_results['training_time']:.2f}s")
        
        logger.info(f"Advanced training completed in {total_training_time:.2f}s total")
        
        # Final evaluation
        self.run_comprehensive_evaluation(adapter)
        
        return {
            'total_training_time': total_training_time,
            'curriculum_stages': len(curriculum_datasets),
            'final_model': adapter,
            'model_manager': self.model_manager
        }
    
    def run_comprehensive_evaluation(self, adapter: NeurosymbolicAdapter):
        """Run comprehensive evaluation of trained model"""
        logger.info("Running comprehensive evaluation")
        
        eval_cfg = self.config['evaluation']
        
        # Test queries of varying complexity
        test_queries = [
            ("Simple SELECT", "Find all customers", "customers (id, name, email)"),
            ("JOIN query", "Find customers with orders", "customers (id, name, email), orders (id, customer_id, amount)"),
            ("Aggregation", "Count orders per customer", "customers (id, name, email), orders (id, customer_id, amount)"),
            ("Complex query", "Find customers with above average order amounts", "customers (id, name, email), orders (id, customer_id, amount)")
        ]
        
        results = []
        for query_type, instruction, schema in test_queries:
            result = adapter.generate_sql(instruction)
            
            results.append({
                'type': query_type,
                'instruction': instruction,
                'sql': result['sql'],
                'confidence': result['confidence'],
                'facts_extracted': len(result.get('extracted_facts', [])),
                'has_symbolic_context': result.get('symbolic_context') is not None
            })
            
            logger.info(f"{query_type}: SQL='{result['sql']}', Confidence={result['confidence']:.3f}")
        
        # Calculate aggregate metrics
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        avg_facts = sum(r['facts_extracted'] for r in results) / len(results)
        
        logger.info(f"Evaluation complete: Avg confidence={avg_confidence:.3f}, Avg facts={avg_facts:.1f}")
        
        return results

def main():
    """Run advanced training example"""
    print("🚀 Advanced Training Example for Neurosymbolic SQL Adapter")
    print("=" * 70)
    
    # Configuration files to demonstrate
    config_files = [
        "../training_configs/advanced_training_config.yaml",
        "../training_configs/production_config.yaml", 
        "../training_configs/research_config.yaml"
    ]
    
    for config_file in config_files:
        config_path = Path(__file__).parent / config_file
        
        if config_path.exists():
            print(f"\n📋 Running training with {config_path.name}")
            print("-" * 50)
            
            try:
                # Create training pipeline
                pipeline = AdvancedTrainingPipeline(str(config_path))
                
                # Run training (with smaller dataset for demo)
                results = pipeline.run_advanced_training()
                
                print(f"✅ Training completed successfully!")
                print(f"   Total time: {results['total_training_time']:.2f}s")
                print(f"   Curriculum stages: {results['curriculum_stages']}")
                print(f"   Model ready for deployment")
                
            except Exception as e:
                print(f"❌ Training failed: {e}")
        else:
            print(f"⚠️  Configuration file not found: {config_path}")
    
    print(f"\n🎉 Advanced training examples completed!")

if __name__ == "__main__":
    main()