#!/usr/bin/env python3
"""
Comprehensive Test Suite for Neural Adapter Components

Tests all neural adapter components including NeurosymbolicAdapter,
BridgeLayer, ConfidenceEstimator, FactExtractor, AdapterTrainer, and ModelManager.
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any

# Import components to test
from src.adapters.neurosymbolic_adapter import NeurosymbolicAdapter, AdapterConfig
from src.adapters.bridge_layer import BridgeLayer, create_llama_bridge, SymbolicContext
from src.adapters.confidence_estimator import (
    ConfidenceEstimator, create_llama_confidence_estimator, 
    ConfidenceMethod, ConfidenceOutput
)
from src.adapters.fact_extractor import (
    FactExtractor, create_llama_fact_extractor, extract_facts_from_sql,
    FactType, ExtractedFact, FactExtractionResult
)
from src.adapters.adapter_trainer import (
    AdapterTrainer, TrainingConfig, create_mock_dataset, NeurosymbolicLoss
)
from src.adapters.model_manager import (
    ModelManager, ModelConfig, ModelType, DeviceType, 
    create_development_config, create_llama_8b_config
)


class TestNeurosymbolicAdapter:
    """Test cases for NeurosymbolicAdapter"""
    
    @pytest.fixture
    def adapter_config(self):
        """Create test adapter configuration"""
        return AdapterConfig(
            lora_r=8,
            lora_alpha=16,
            bridge_dim=256,
            symbolic_dim=128
        )
    
    @pytest.fixture
    def adapter(self, adapter_config):
        """Create test adapter"""
        return NeurosymbolicAdapter(
            base_model_name="test/model",
            config=adapter_config
        )
    
    def test_adapter_initialization(self, adapter, adapter_config):
        """Test adapter initialization"""
        assert adapter.config.lora_r == adapter_config.lora_r
        assert adapter.config.lora_alpha == adapter_config.lora_alpha
        assert adapter.base_model_name == "test/model"
        assert not adapter.is_trained
        
    def test_adapter_forward_pass(self, adapter):
        """Test adapter forward pass"""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones_like(input_ids)
        
        outputs = adapter.forward(input_ids, attention_mask)
        
        assert outputs.logits.shape == (2, 10, 32000)  # Batch, seq, vocab
        assert outputs.symbolic_context is None  # No bridge layer yet
        assert outputs.confidence_scores is None  # No confidence estimator yet
        assert outputs.extracted_facts is None  # No fact extractor yet
    
    def test_adapter_sql_generation(self, adapter):
        """Test SQL generation"""
        result = adapter.generate_sql("Find all customers")
        
        assert "sql" in result
        assert "confidence" in result
        assert isinstance(result["sql"], str)
        assert isinstance(result["confidence"], float)
        assert 0 <= result["confidence"] <= 1
    
    def test_adapter_component_integration(self, adapter):
        """Test component integration"""
        # Initially no components
        status = adapter.get_status()
        assert not status["components"]["bridge_layer"]
        assert not status["components"]["confidence_estimator"]
        assert not status["components"]["fact_extractor"]
        
        # Add bridge layer
        bridge = create_llama_bridge()
        adapter.set_bridge_layer(bridge)
        status = adapter.get_status()
        assert status["components"]["bridge_layer"]
        
        # Add confidence estimator
        confidence_estimator = create_llama_confidence_estimator(
            methods=[ConfidenceMethod.ENTROPY]
        )
        adapter.set_confidence_estimator(confidence_estimator)
        status = adapter.get_status()
        assert status["components"]["confidence_estimator"]
        
        # Add fact extractor
        fact_extractor = create_llama_fact_extractor()
        adapter.set_fact_extractor(fact_extractor)
        status = adapter.get_status()
        assert status["components"]["fact_extractor"]
    
    def test_adapter_save_load(self, adapter):
        """Test adapter save/load functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_adapter"
            
            # Save adapter
            adapter.save_adapter(save_path)
            
            # Verify files exist
            assert (save_path / "adapter_config.json").exists()
            
            # Load adapter
            adapter.load_adapter(save_path)
            
            # Verify loading worked (no exceptions)
            assert True


class TestBridgeLayer:
    """Test cases for BridgeLayer"""
    
    @pytest.fixture
    def bridge_layer(self):
        """Create test bridge layer"""
        return BridgeLayer(
            neural_dim=1024,
            symbolic_dim=256,
            bridge_dim=128,
            num_layers=2
        )
    
    def test_bridge_initialization(self, bridge_layer):
        """Test bridge layer initialization"""
        assert bridge_layer.neural_dim == 1024
        assert bridge_layer.symbolic_dim == 256
        assert bridge_layer.bridge_dim == 128
        assert len(bridge_layer.transformer_layers) == 2
        assert len(bridge_layer.concept_names) >= 100  # At least 100 concepts
    
    def test_bridge_forward_pass(self, bridge_layer):
        """Test bridge layer forward pass"""
        neural_input = torch.randn(2, 8, 1024)  # Batch, seq, neural_dim
        
        symbolic_context = bridge_layer.forward(neural_input)
        
        assert isinstance(symbolic_context, SymbolicContext)
        assert symbolic_context.embeddings.shape == (2, 8, 256)  # Batch, seq, symbolic_dim
        assert len(symbolic_context.facts) == 2  # One per batch
        assert symbolic_context.attention_weights is not None
        assert symbolic_context.reasoning_trace is not None
    
    def test_bridge_fact_extraction(self, bridge_layer):
        """Test fact extraction from bridge layer"""
        neural_input = torch.randn(1, 10, 1024)
        symbolic_context = bridge_layer.forward(neural_input)
        
        facts = symbolic_context.facts[0]
        assert isinstance(facts, list)
        assert len(facts) > 0
        
        # Check fact format
        for fact in facts[:3]:  # Check first few facts
            assert "confidence=" in fact
            assert isinstance(fact, str)
    
    def test_bridge_concept_similarities(self, bridge_layer):
        """Test concept similarity computation"""
        query_embedding = torch.randn(256)  # symbolic_dim
        
        similarities = bridge_layer.get_concept_similarities(query_embedding)
        
        assert isinstance(similarities, dict)
        assert len(similarities) <= 100  # Number of concepts
        
        for concept, score in similarities.items():
            assert isinstance(concept, str)
            assert isinstance(score, float)
            assert -1 <= score <= 1  # Cosine similarity range
    
    def test_bridge_explanation(self, bridge_layer):
        """Test symbolic transformation explanation"""
        neural_input = torch.randn(1, 5, 1024)
        
        explanation = bridge_layer.explain_symbolic_transformation(neural_input)
        
        assert isinstance(explanation, dict)
        assert "input_shape" in explanation
        assert "symbolic_shape" in explanation
        assert "top_concepts" in explanation
        assert "confidence_distribution" in explanation
        
        assert explanation["input_shape"] == [1, 5, 1024]
        assert explanation["symbolic_shape"] == [1, 5, 256]


class TestConfidenceEstimator:
    """Test cases for ConfidenceEstimator"""
    
    @pytest.fixture
    def confidence_estimator(self):
        """Create test confidence estimator"""
        return ConfidenceEstimator(
            vocab_size=1000,
            hidden_dim=512,
            symbolic_dim=256,
            methods=[ConfidenceMethod.ENTROPY, ConfidenceMethod.TEMPERATURE_SCALING]
        )
    
    def test_confidence_initialization(self, confidence_estimator):
        """Test confidence estimator initialization"""
        assert confidence_estimator.vocab_size == 1000
        assert confidence_estimator.hidden_dim == 512
        assert confidence_estimator.symbolic_dim == 256
        assert len(confidence_estimator.methods) == 2
        assert "entropy" in confidence_estimator.estimators
        assert "temperature" in confidence_estimator.estimators
    
    def test_confidence_forward_pass(self, confidence_estimator):
        """Test confidence estimator forward pass"""
        batch_size, seq_len, vocab_size = 2, 8, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size)
        hidden_states = torch.randn(batch_size, seq_len, 512)
        symbolic_embeddings = torch.randn(batch_size, seq_len, 256)
        
        output = confidence_estimator.forward(
            logits=logits,
            hidden_states=hidden_states,
            symbolic_embeddings=symbolic_embeddings
        )
        
        assert isinstance(output, ConfidenceOutput)
        assert 0 <= output.overall_confidence <= 1
        assert output.token_confidences.shape == (batch_size, seq_len)
        assert output.uncertainty_estimate >= 0
        assert output.calibration_score is not None
        assert output.method_scores is not None
        assert output.explanation is not None
    
    def test_confidence_individual_methods(self):
        """Test individual confidence methods"""
        from src.adapters.confidence_estimator import (
            EntropyConfidenceEstimator, TemperatureScalingEstimator
        )
        
        logits = torch.randn(2, 5, 1000)
        
        # Test entropy estimator
        entropy_est = EntropyConfidenceEstimator(vocab_size=1000)
        entropy_conf = entropy_est(logits)
        assert entropy_conf.shape == (2, 5)
        assert torch.all(entropy_conf >= 0) and torch.all(entropy_conf <= 1)
        
        # Test temperature scaling
        temp_est = TemperatureScalingEstimator()
        temp_conf = temp_est(logits)
        assert temp_conf.shape == (2, 5)
        assert torch.all(temp_conf >= 0) and torch.all(temp_conf <= 1)
    
    def test_confidence_analysis(self, confidence_estimator):
        """Test uncertainty analysis"""
        logits = torch.randn(1, 10, 1000)
        hidden_states = torch.randn(1, 10, 512)
        
        analysis = confidence_estimator.analyze_uncertainty_sources(
            logits, hidden_states
        )
        
        assert isinstance(analysis, dict)
        assert "overall_confidence" in analysis
        assert "uncertainty_estimate" in analysis
        assert "method_contributions" in analysis
        assert "confidence_distribution" in analysis
        assert "method_weights" in analysis


class TestFactExtractor:
    """Test cases for FactExtractor"""
    
    @pytest.fixture
    def fact_extractor(self):
        """Create test fact extractor"""
        return FactExtractor(
            hidden_dim=512,
            extraction_threshold=0.4,
            max_facts_per_query=20,
            enable_pattern_matching=True
        )
    
    def test_fact_extractor_initialization(self, fact_extractor):
        """Test fact extractor initialization"""
        assert fact_extractor.hidden_dim == 512
        assert fact_extractor.extraction_threshold == 0.4
        assert fact_extractor.max_facts_per_query == 20
        assert fact_extractor.enable_pattern_matching
        assert fact_extractor.pattern_matcher is not None
    
    def test_pattern_based_extraction(self):
        """Test pattern-based fact extraction"""
        sql_query = "SELECT c.name, COUNT(o.id) FROM customers c JOIN orders o ON c.id = o.customer_id WHERE c.active = 1"
        
        facts = extract_facts_from_sql(sql_query)
        
        assert len(facts) > 0
        assert all(isinstance(fact, ExtractedFact) for fact in facts)
        
        # Check for expected fact types
        fact_strings = [fact.fact_string for fact in facts]
        assert any("query_type(select)" in f for f in fact_strings)
        assert any("query_references_table" in f for f in fact_strings)
        assert any("query_uses_aggregate" in f for f in fact_strings)
    
    def test_neural_fact_extraction(self, fact_extractor):
        """Test neural fact extraction"""
        hidden_states = torch.randn(1, 12, 512)
        sql_text = "SELECT * FROM users WHERE age > 18"
        
        result = fact_extractor.forward(
            hidden_states=hidden_states,
            input_text=sql_text
        )
        
        assert isinstance(result, FactExtractionResult)
        assert len(result.facts) <= fact_extractor.max_facts_per_query
        assert 0 <= result.extraction_confidence <= 1
        assert result.processing_time > 0
        
        # Check fact counts by type
        assert isinstance(result.fact_counts, dict)
        assert all(isinstance(count, int) for count in result.fact_counts.values())
        
        # Check metadata
        assert "neural_facts" in result.metadata
        assert "pattern_facts" in result.metadata
        assert "total_candidates" in result.metadata
    
    def test_fact_validation(self, fact_extractor):
        """Test fact validation"""
        # Create test facts
        test_facts = [
            ExtractedFact("table_exists(users)", FactType.SCHEMA_FACT, 0.9),
            ExtractedFact("query_type(select)", FactType.QUERY_FACT, 0.8),
            ExtractedFact("column_exists(users, name)", FactType.SCHEMA_FACT, 0.7)
        ]
        
        validation_result = fact_extractor.validate_extracted_facts(test_facts)
        
        assert isinstance(validation_result, dict)
        assert "is_consistent" in validation_result
        assert "inconsistencies" in validation_result
        assert "warnings" in validation_result
        assert "recommendations" in validation_result
    
    def test_fact_statistics(self, fact_extractor):
        """Test fact statistics computation"""
        test_facts = [
            ExtractedFact("fact1", FactType.QUERY_FACT, 0.9),
            ExtractedFact("fact2", FactType.SCHEMA_FACT, 0.7),
            ExtractedFact("fact3", FactType.QUERY_FACT, 0.6)
        ]
        
        stats = fact_extractor.get_fact_statistics(test_facts)
        
        assert stats["total_facts"] == 3
        assert 0 <= stats["average_confidence"] <= 1
        assert "confidence_distribution" in stats
        assert "fact_types" in stats
        assert "top_facts" in stats


class TestAdapterTrainer:
    """Test cases for AdapterTrainer"""
    
    @pytest.fixture
    def training_config(self):
        """Create test training configuration"""
        return TrainingConfig(
            num_epochs=1,
            batch_size=2,
            learning_rate=1e-4,
            lora_r=8,
            eval_steps=5,
            logging_steps=2,
            output_dir="./test_training_output"
        )
    
    @pytest.fixture
    def trainer(self, training_config):
        """Create test trainer"""
        return AdapterTrainer(training_config)
    
    def test_trainer_initialization(self, trainer, training_config):
        """Test trainer initialization"""
        assert trainer.config.num_epochs == training_config.num_epochs
        assert trainer.config.batch_size == training_config.batch_size
        assert trainer.state.epoch == 0
        assert trainer.state.global_step == 0
        assert trainer.model is not None
    
    def test_mock_dataset_creation(self):
        """Test mock dataset creation"""
        dataset = create_mock_dataset(50)
        
        assert len(dataset) == 50
        
        sample = dataset[0]
        assert "input_ids" in sample
        assert "labels" in sample
        assert "attention_mask" in sample
        assert "instruction" in sample
        assert "sql" in sample
        assert "schema" in sample
    
    def test_loss_function(self, training_config):
        """Test neurosymbolic loss function"""
        loss_fn = NeurosymbolicLoss(training_config)
        
        # Create mock outputs
        batch_size, seq_len, vocab_size = 2, 10, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Mock model outputs
        from src.adapters.neurosymbolic_adapter import NeurosymbolicOutput
        outputs = NeurosymbolicOutput(logits=logits)
        
        loss_dict = loss_fn(outputs, labels)
        
        assert "base_loss" in loss_dict
        assert "total_loss" in loss_dict
        assert loss_dict["total_loss"].requires_grad
    
    def test_trainer_setup(self, trainer):
        """Test trainer setup"""
        train_dataset = create_mock_dataset(10)
        eval_dataset = create_mock_dataset(5)
        
        trainer.setup_training(train_dataset, eval_dataset)
        
        assert trainer.train_loader is not None
        assert trainer.eval_loader is not None
        assert trainer.optimizer is not None
        assert len(trainer.train_loader) == 5  # 10 samples / 2 batch_size
    
    def test_training_summary(self, trainer):
        """Test training summary generation"""
        summary = trainer.get_training_summary()
        
        assert "config" in summary
        assert "model_info" in summary
        assert "training_state" in summary
        assert "performance" in summary
        
        assert "trainable_parameters" in summary["model_info"]
        assert "components" in summary["model_info"]


class TestModelManager:
    """Test cases for ModelManager"""
    
    @pytest.fixture
    def model_config(self):
        """Create test model configuration"""
        return ModelConfig(
            model_type=ModelType.LLAMA_8B,
            model_name="test/model",
            device=DeviceType.CPU,  # Use CPU for testing
            lora_r=8,
            bridge_dim=256,
            symbolic_dim=128
        )
    
    @pytest.fixture
    def model_manager(self, model_config):
        """Create test model manager"""
        return ModelManager(model_config)
    
    def test_manager_initialization(self, model_manager, model_config):
        """Test model manager initialization"""
        assert model_manager.config.model_type == model_config.model_type
        assert model_manager.device.type == "cpu"
        assert len(model_manager.model_configs) == 5  # Number of supported model types
    
    def test_model_loading_and_info(self, model_manager, model_config):
        """Test model loading and info retrieval"""
        model_id = "test_model"
        
        # Load model
        adapter = model_manager.load_model(model_id, model_config)
        assert adapter is not None
        assert model_id in model_manager.models
        
        # Get model info
        info = model_manager.get_model_info(model_id)
        assert info is not None
        assert info.model_type == model_config.model_type
        assert info.device == "cpu"
        assert info.parameter_count["total"] > 0
        assert info.load_time > 0
        
        # Check components
        assert len(info.component_status) == 4
        assert all(info.component_status.values())  # All components should be active
    
    def test_model_save_load(self, model_manager, model_config):
        """Test model save/load functionality"""
        model_id = "save_test_model"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "saved_model"
            
            # Load and save model
            adapter = model_manager.load_model(model_id, model_config)
            success = model_manager.save_model(model_id, save_path)
            assert success
            
            # Verify files exist
            assert save_path.exists()
            assert (save_path / "model_info.json").exists()
            
            # Unload and reload
            model_manager.unload_model(model_id)
            assert model_id not in model_manager.models
            
            reloaded_adapter = model_manager.load_saved_model(model_id, save_path)
            assert reloaded_adapter is not None
            assert model_id in model_manager.models
    
    def test_system_status(self, model_manager, model_config):
        """Test system status reporting"""
        # Load a model first
        model_manager.load_model("status_test", model_config)
        
        status = model_manager.get_system_status()
        
        assert "device" in status
        assert "memory_usage" in status
        assert "loaded_models" in status
        assert "model_list" in status
        assert "model_info" in status
        
        assert status["loaded_models"] == 1
        assert "status_test" in status["model_list"]
    
    def test_model_configurations(self, model_manager):
        """Test different model configurations"""
        # Test Llama 8B config
        llama_config = create_llama_8b_config()
        assert llama_config.model_type == ModelType.LLAMA_8B
        assert llama_config.lora_r == 16
        
        # Test development config
        dev_config = create_development_config()
        assert dev_config.model_type == ModelType.LLAMA_8B
        assert dev_config.lora_r == 8  # Smaller for development
        assert dev_config.load_in_4bit
    
    def test_cleanup(self, model_manager, model_config):
        """Test cleanup functionality"""
        # Load multiple models
        model_manager.load_model("cleanup_test_1", model_config)
        model_manager.load_model("cleanup_test_2", model_config)
        
        assert len(model_manager.list_models()) == 2
        
        # Cleanup
        model_manager.cleanup()
        
        assert len(model_manager.list_models()) == 0


class TestIntegration:
    """Integration tests for all components working together"""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end neurosymbolic pipeline"""
        # Create model manager with CPU config
        config = ModelConfig(
            model_type=ModelType.LLAMA_8B,
            device=DeviceType.CPU,
            lora_r=8,
            bridge_dim=128,
            symbolic_dim=64,
            enable_bridge=True,
            enable_confidence=True,
            enable_fact_extraction=True
        )
        
        manager = ModelManager(config)
        
        # Load model with all components
        model_id = "integration_test"
        adapter = manager.load_model(model_id, config)
        
        # Verify all components are active
        status = adapter.get_status()
        assert all(status["components"].values())
        
        # Test forward pass with all components
        input_ids = torch.randint(0, 1000, (1, 8))
        outputs = adapter.forward(input_ids)
        
        assert outputs.logits is not None
        assert outputs.symbolic_context is not None
        assert outputs.confidence_scores is not None
        assert outputs.extracted_facts is not None
        
        # Test SQL generation
        result = manager.generate_sql(model_id, "Find all active users")
        assert result is not None
        assert "sql" in result
        assert "confidence" in result
        assert "model_id" in result
        
        # Cleanup
        manager.cleanup()
    
    def test_component_compatibility(self):
        """Test compatibility between different components"""
        # Create components with compatible dimensions
        bridge = BridgeLayer(
            neural_dim=512,
            symbolic_dim=256,
            bridge_dim=128
        )
        
        confidence_estimator = ConfidenceEstimator(
            vocab_size=1000,
            hidden_dim=512,
            symbolic_dim=256,
            methods=[ConfidenceMethod.ENTROPY]
        )
        
        fact_extractor = FactExtractor(
            hidden_dim=256,  # Match symbolic_dim
            extraction_threshold=0.5,
            max_facts_per_query=20
        )
        
        # Test data flow compatibility
        neural_input = torch.randn(1, 10, 512)
        
        # Bridge layer
        symbolic_context = bridge.forward(neural_input)
        assert symbolic_context.embeddings.shape == (1, 10, 256)
        
        # Confidence estimation (needs logits)
        logits = torch.randn(1, 10, 1000)
        confidence_output = confidence_estimator.forward(
            logits=logits,
            hidden_states=neural_input,
            symbolic_embeddings=symbolic_context.embeddings
        )
        assert 0 <= confidence_output.overall_confidence <= 1
        
        # Fact extraction (note: fact_extractor expects neural input, not symbolic)
        fact_result = fact_extractor.forward(
            hidden_states=neural_input[:, :, :256],  # Match fact_extractor hidden_dim
            symbolic_embeddings=symbolic_context.embeddings
        )
        assert len(fact_result.facts) <= fact_extractor.max_facts_per_query


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])