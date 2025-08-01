#!/usr/bin/env python3
"""
Model Manager

Centralized management system for loading, configuring, and managing
neurosymbolic SQL adapters and their base models.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml
from enum import Enum
import time

# Import neurosymbolic components
from .neurosymbolic_adapter import NeurosymbolicAdapter, AdapterConfig
from .bridge_layer import BridgeLayer, create_llama_bridge
from .confidence_estimator import ConfidenceEstimator, create_llama_confidence_estimator, ConfidenceMethod
from .fact_extractor import FactExtractor, create_llama_fact_extractor
from .adapter_trainer import AdapterTrainer, TrainingConfig


class ModelType(Enum):
    """Supported model types"""
    LLAMA_7B = "llama-7b"
    LLAMA_8B = "llama-8b"
    LLAMA_13B = "llama-13b"
    LLAMA_70B = "llama-70b"
    MISTRAL_7B = "mistral-7b"
    CODELLAMA_7B = "codellama-7b"
    CODELLAMA_13B = "codellama-13b"
    CUSTOM = "custom"


class DeviceType(Enum):
    """Device types for model deployment"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal Performance Shaders
    AUTO = "auto"


@dataclass
class ModelConfig:
    """Comprehensive model configuration"""
    # Base model settings
    model_type: ModelType = ModelType.LLAMA_8B
    model_name: str = "unsloth/llama-3.1-8b-instruct-bnb-4bit"
    device: DeviceType = DeviceType.AUTO
    torch_dtype: str = "float16"  # "float16", "bfloat16", "float32"
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Neurosymbolic configuration
    bridge_dim: int = 512
    symbolic_dim: int = 256
    enable_bridge: bool = True
    enable_confidence: bool = True
    enable_fact_extraction: bool = True
    
    # Confidence estimation settings
    confidence_methods: List[ConfidenceMethod] = field(default_factory=lambda: [
        ConfidenceMethod.ENTROPY,
        ConfidenceMethod.TEMPERATURE_SCALING
    ])
    
    # Fact extraction settings
    fact_extraction_threshold: float = 0.5
    max_facts_per_query: int = 50
    enable_pattern_matching: bool = True
    
    # Memory and performance settings
    gradient_checkpointing: bool = False
    use_cache: bool = True
    max_memory: Optional[Dict[str, str]] = None
    offload_folder: Optional[str] = None
    
    # Quantization settings
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class ModelInfo:
    """Information about loaded model"""
    model_type: ModelType
    model_name: str
    device: str
    memory_usage: Dict[str, float]
    parameter_count: Dict[str, int]
    component_status: Dict[str, bool]
    load_time: float
    config: ModelConfig


class ModelManager:
    """
    Centralized Model Manager
    
    Handles loading, configuration, and management of neurosymbolic SQL adapters.
    Provides high-level interface for model operations.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize ModelManager
        
        Args:
            config: Model configuration (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or ModelConfig()
        
        # Model registry
        self.models: Dict[str, NeurosymbolicAdapter] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        
        # Device management
        self.device = self._setup_device()
        
        # Model configurations for different types
        self.model_configs = self._initialize_model_configs()
        
        self.logger.info(f"ModelManager initialized with device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup optimal device for model loading"""
        if self.config.device == DeviceType.AUTO:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device.value)
        
        self.logger.info(f"Selected device: {device}")
        return device
    
    def _initialize_model_configs(self) -> Dict[ModelType, Dict[str, Any]]:
        """Initialize configurations for different model types"""
        configs = {
            ModelType.LLAMA_7B: {
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_hidden_layers": 32,
                "vocab_size": 32000,
                "max_position_embeddings": 4096,
                "recommended_batch_size": 4
            },
            ModelType.LLAMA_8B: {
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_hidden_layers": 32,
                "vocab_size": 32000,
                "max_position_embeddings": 8192,
                "recommended_batch_size": 4
            },
            ModelType.LLAMA_13B: {
                "hidden_size": 5120,
                "num_attention_heads": 40,
                "num_hidden_layers": 40,
                "vocab_size": 32000,
                "max_position_embeddings": 4096,
                "recommended_batch_size": 2
            },
            ModelType.LLAMA_70B: {
                "hidden_size": 8192,
                "num_attention_heads": 64,
                "num_hidden_layers": 80,
                "vocab_size": 32000,
                "max_position_embeddings": 4096,
                "recommended_batch_size": 1
            },
            ModelType.MISTRAL_7B: {
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_hidden_layers": 32,
                "vocab_size": 32000,
                "max_position_embeddings": 32768,
                "recommended_batch_size": 4
            }
        }
        
        return configs
    
    def load_model(self, model_id: str, config: Optional[ModelConfig] = None) -> NeurosymbolicAdapter:
        """
        Load neurosymbolic adapter model
        
        Args:
            model_id: Unique identifier for the model
            config: Model configuration (optional)
            
        Returns:
            Loaded NeurosymbolicAdapter
        """
        start_time = time.time()
        
        if model_id in self.models:
            self.logger.info(f"Model {model_id} already loaded")
            return self.models[model_id]
        
        effective_config = config or self.config
        self.logger.info(f"Loading model {model_id} with type {effective_config.model_type}")
        
        # Create adapter configuration
        adapter_config = self._create_adapter_config(effective_config)
        
        # Initialize neurosymbolic adapter
        adapter = NeurosymbolicAdapter(
            base_model_name=effective_config.model_name,
            config=adapter_config
        )
        
        # Initialize neurosymbolic components
        self._initialize_components(adapter, effective_config)
        
        # Move to device
        adapter = adapter.to(self.device)
        
        # Store model and info
        load_time = time.time() - start_time
        self.models[model_id] = adapter
        self.model_info[model_id] = self._create_model_info(adapter, effective_config, load_time)
        
        self.logger.info(f"Model {model_id} loaded successfully in {load_time:.2f}s")
        return adapter
    
    def _create_adapter_config(self, config: ModelConfig) -> AdapterConfig:
        """Create AdapterConfig from ModelConfig"""
        return AdapterConfig(
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bridge_dim=config.bridge_dim,
            symbolic_dim=config.symbolic_dim,
            confidence_dim=128,
            enable_uncertainty_estimation=config.enable_confidence,
            fact_extraction_dim=256,
            max_facts_per_query=config.max_facts_per_query,
            fact_threshold=config.fact_extraction_threshold
        )
    
    def _initialize_components(self, adapter: NeurosymbolicAdapter, config: ModelConfig):
        """Initialize neurosymbolic components for adapter"""
        model_config = self.model_configs.get(config.model_type, self.model_configs[ModelType.LLAMA_8B])
        hidden_size = model_config["hidden_size"]
        
        # Bridge layer
        if config.enable_bridge:
            if config.model_type in [ModelType.LLAMA_7B, ModelType.LLAMA_8B, ModelType.LLAMA_13B, ModelType.LLAMA_70B]:
                bridge = create_llama_bridge()
            else:
                bridge = BridgeLayer(
                    neural_dim=hidden_size,
                    symbolic_dim=config.symbolic_dim,
                    bridge_dim=config.bridge_dim
                )
            adapter.set_bridge_layer(bridge)
        
        # Confidence estimator
        if config.enable_confidence:
            if config.model_type in [ModelType.LLAMA_7B, ModelType.LLAMA_8B, ModelType.LLAMA_13B, ModelType.LLAMA_70B]:
                confidence_estimator = create_llama_confidence_estimator(methods=config.confidence_methods)
            else:
                confidence_estimator = ConfidenceEstimator(
                    vocab_size=model_config["vocab_size"],
                    hidden_dim=hidden_size,
                    symbolic_dim=config.symbolic_dim,
                    methods=config.confidence_methods
                )
            adapter.set_confidence_estimator(confidence_estimator)
        
        # Fact extractor
        if config.enable_fact_extraction:
            if config.model_type in [ModelType.LLAMA_7B, ModelType.LLAMA_8B, ModelType.LLAMA_13B, ModelType.LLAMA_70B]:
                fact_extractor = create_llama_fact_extractor(
                    extraction_threshold=config.fact_extraction_threshold,
                    max_facts_per_query=config.max_facts_per_query,
                    enable_pattern_matching=config.enable_pattern_matching
                )
            else:
                fact_extractor = FactExtractor(
                    hidden_dim=hidden_size,
                    extraction_threshold=config.fact_extraction_threshold,
                    max_facts_per_query=config.max_facts_per_query,
                    enable_pattern_matching=config.enable_pattern_matching
                )
            adapter.set_fact_extractor(fact_extractor)
    
    def _create_model_info(self, adapter: NeurosymbolicAdapter, config: ModelConfig, load_time: float) -> ModelInfo:
        """Create ModelInfo for loaded adapter"""
        # Get memory usage
        memory_usage = self._get_memory_usage()
        
        # Get parameter counts
        parameter_count = {
            'total': sum(p.numel() for p in adapter.parameters()),
            'trainable': adapter.get_trainable_parameters()
        }
        
        # Get component status
        component_status = adapter.get_status()['components']
        
        return ModelInfo(
            model_type=config.model_type,
            model_name=config.model_name,
            device=str(self.device),
            memory_usage=memory_usage,
            parameter_count=parameter_count,
            component_status=component_status,
            load_time=load_time,
            config=config
        )
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        memory_usage = {'cpu': 0.0, 'gpu': 0.0}
        
        # CPU memory (approximation)
        import psutil
        process = psutil.Process()
        memory_usage['cpu'] = process.memory_info().rss / 1024**2  # MB
        
        # GPU memory
        if self.device.type == 'cuda':
            memory_usage['gpu'] = torch.cuda.memory_allocated() / 1024**2  # MB
        
        return memory_usage
    
    def unload_model(self, model_id: str) -> bool:
        """
        Unload model from memory
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if successful
        """
        if model_id not in self.models:
            self.logger.warning(f"Model {model_id} not found")
            return False
        
        # Remove from registry
        del self.models[model_id]
        del self.model_info[model_id]
        
        # Clear GPU memory if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        self.logger.info(f"Model {model_id} unloaded")
        return True
    
    def get_model(self, model_id: str) -> Optional[NeurosymbolicAdapter]:
        """Get loaded model by ID"""
        return self.models.get(model_id)
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get model information by ID"""
        return self.model_info.get(model_id)
    
    def list_models(self) -> List[str]:
        """List all loaded model IDs"""
        return list(self.models.keys())
    
    def save_model(self, model_id: str, save_path: Union[str, Path]) -> bool:
        """
        Save model to disk
        
        Args:
            model_id: Model identifier
            save_path: Path to save location
            
        Returns:
            True if successful
        """
        if model_id not in self.models:
            self.logger.error(f"Model {model_id} not found")
            return False
        
        try:
            adapter = self.models[model_id]
            adapter.save_adapter(save_path)
            
            # Save model info
            info_path = Path(save_path) / "model_info.json"
            with open(info_path, 'w') as f:
                info_dict = {
                    'model_type': self.model_info[model_id].model_type.value,
                    'model_name': self.model_info[model_id].model_name,
                    'device': self.model_info[model_id].device,
                    'parameter_count': self.model_info[model_id].parameter_count,
                    'component_status': self.model_info[model_id].component_status,
                    'load_time': self.model_info[model_id].load_time
                }
                json.dump(info_dict, f, indent=2)
            
            self.logger.info(f"Model {model_id} saved to {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model {model_id}: {e}")
            return False
    
    def load_saved_model(self, model_id: str, load_path: Union[str, Path]) -> Optional[NeurosymbolicAdapter]:
        """
        Load model from disk
        
        Args:
            model_id: Model identifier
            load_path: Path to saved model
            
        Returns:
            Loaded adapter or None if failed
        """
        try:
            # Load model info
            info_path = Path(load_path) / "model_info.json"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    info_dict = json.load(f)
                
                # Create config from saved info
                config = ModelConfig(
                    model_type=ModelType(info_dict['model_type']),
                    model_name=info_dict['model_name']
                )
            else:
                config = self.config
            
            # Load model
            adapter = self.load_model(model_id, config)
            adapter.load_adapter(load_path)
            
            self.logger.info(f"Model {model_id} loaded from {load_path}")
            return adapter
            
        except Exception as e:
            self.logger.error(f"Error loading model from {load_path}: {e}")
            return None
    
    def create_trainer(self, model_id: str, training_config: TrainingConfig) -> Optional[AdapterTrainer]:
        """
        Create trainer for model
        
        Args:
            model_id: Model identifier
            training_config: Training configuration
            
        Returns:
            AdapterTrainer or None if model not found
        """
        if model_id not in self.models:
            self.logger.error(f"Model {model_id} not found")
            return None
        
        adapter = self.models[model_id]
        trainer = AdapterTrainer(training_config, adapter)
        
        self.logger.info(f"Trainer created for model {model_id}")
        return trainer
    
    def generate_sql(self, model_id: str, instruction: str, schema: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Generate SQL using specified model
        
        Args:
            model_id: Model identifier
            instruction: Natural language instruction
            schema: Database schema (optional)
            
        Returns:
            Generation result or None if failed
        """
        if model_id not in self.models:
            self.logger.error(f"Model {model_id} not found")
            return None
        
        try:
            adapter = self.models[model_id]
            result = adapter.generate_sql(instruction)
            
            # Add schema if provided
            if schema:
                result['schema'] = schema
                
            # Add model info
            result['model_id'] = model_id
            result['model_type'] = self.model_info[model_id].model_type.value
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating SQL with model {model_id}: {e}")
            return None
    
    def benchmark_model(self, model_id: str, num_queries: int = 10) -> Optional[Dict[str, Any]]:
        """
        Benchmark model performance
        
        Args:
            model_id: Model identifier
            num_queries: Number of test queries
            
        Returns:
            Benchmark results
        """
        if model_id not in self.models:
            self.logger.error(f"Model {model_id} not found")
            return None
        
        adapter = self.models[model_id]
        
        # Sample queries for benchmarking
        test_queries = [
            "Find all customers with orders",
            "Get total sales by product category",
            "List top 10 customers by revenue",
            "Show monthly sales trends",
            "Find products with low inventory"
        ] * (num_queries // 5 + 1)
        
        start_time = time.time()
        results = []
        
        for i, query in enumerate(test_queries[:num_queries]):
            query_start = time.time()
            result = adapter.generate_sql(query)
            query_time = time.time() - query_start
            
            results.append({
                'query': query,
                'sql': result['sql'],
                'confidence': result['confidence'],
                'time': query_time
            })
        
        total_time = time.time() - start_time
        avg_time = total_time / num_queries
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        
        benchmark = {
            'model_id': model_id,
            'num_queries': num_queries,
            'total_time': total_time,
            'average_time_per_query': avg_time,
            'average_confidence': avg_confidence,
            'queries_per_second': num_queries / total_time,
            'results': results
        }
        
        self.logger.info(f"Benchmark completed for {model_id}: {avg_time:.3f}s/query, {avg_confidence:.3f} confidence")
        return benchmark
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        memory_usage = self._get_memory_usage()
        
        status = {
            'device': str(self.device),
            'memory_usage': memory_usage,
            'loaded_models': len(self.models),
            'model_list': list(self.models.keys()),
            'model_info': {
                model_id: {
                    'type': info.model_type.value,
                    'parameters': info.parameter_count,
                    'components': info.component_status,
                    'device': info.device
                }
                for model_id, info in self.model_info.items()
            }
        }
        
        return status
    
    def cleanup(self):
        """Cleanup all loaded models and free memory"""
        model_ids = list(self.models.keys())
        for model_id in model_ids:
            self.unload_model(model_id)
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        self.logger.info("ModelManager cleanup completed")


# Convenience functions
def create_model_manager(config: Optional[ModelConfig] = None) -> ModelManager:
    """Create ModelManager with configuration"""
    return ModelManager(config)


def create_llama_8b_config() -> ModelConfig:
    """Create configuration for Llama 8B model"""
    return ModelConfig(
        model_type=ModelType.LLAMA_8B,
        model_name="unsloth/llama-3.1-8b-instruct-bnb-4bit",
        lora_r=16,
        lora_alpha=32,
        bridge_dim=512,
        symbolic_dim=256
    )


def create_development_config() -> ModelConfig:
    """Create configuration optimized for development"""
    return ModelConfig(
        model_type=ModelType.LLAMA_8B,
        model_name="unsloth/llama-3.1-8b-instruct-bnb-4bit",
        lora_r=8,  # Smaller for faster training
        lora_alpha=16,
        bridge_dim=256,
        symbolic_dim=128,
        load_in_4bit=True,
        gradient_checkpointing=True
    )


def load_config_from_file(config_path: Union[str, Path]) -> ModelConfig:
    """Load ModelConfig from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert string enums back to enum objects
    if 'model_type' in config_dict:
        config_dict['model_type'] = ModelType(config_dict['model_type'])
    if 'device' in config_dict:
        config_dict['device'] = DeviceType(config_dict['device'])
    if 'confidence_methods' in config_dict:
        config_dict['confidence_methods'] = [ConfidenceMethod(m) for m in config_dict['confidence_methods']]
    
    return ModelConfig(**config_dict)