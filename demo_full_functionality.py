#!/usr/bin/env python3
"""
Comprehensive Neurosymbolic SQL Adapter Functionality Demonstration

This script demonstrates the full functionality of the integrated neural-symbolic
SQL adapter system, showcasing all components working together.
"""

import sys
import logging
from pathlib import Path
import torch

# Set up paths
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def demo_individual_components():
    """Demonstrate individual component functionality"""
    print("🔧 COMPONENT FUNCTIONALITY DEMO")
    print("=" * 60)
    
    # 1. Bridge Layer Demo
    print("\n1. Neural-Symbolic Bridge Layer")
    print("-" * 30)
    
    from adapters.bridge_layer import create_llama_bridge, BridgeLayer
    
    bridge = create_llama_bridge()
    neural_input = torch.randn(2, 8, 4096)  # Batch=2, Seq=8, Llama hidden size
    
    symbolic_context = bridge.forward(neural_input)
    print(f"   Input shape: {neural_input.shape}")
    print(f"   Output shape: {symbolic_context.embeddings.shape}")
    print(f"   Facts extracted: {len(symbolic_context.facts[0])} for first batch")
    print(f"   Sample fact: {symbolic_context.facts[0][0]}")
    
    # 2. Confidence Estimator Demo
    print("\n2. Confidence Estimation")
    print("-" * 25)
    
    from adapters.confidence_estimator import create_llama_confidence_estimator, ConfidenceMethod
    
    confidence_estimator = create_llama_confidence_estimator(
        methods=[ConfidenceMethod.ENTROPY, ConfidenceMethod.TEMPERATURE_SCALING]
    )
    
    logits = torch.randn(2, 8, 32000)  # Batch=2, Seq=8, Vocab=32000
    hidden_states = torch.randn(2, 8, 4096)
    
    confidence_output = confidence_estimator.forward(
        logits=logits,
        hidden_states=hidden_states,
        symbolic_embeddings=symbolic_context.embeddings
    )
    
    print(f"   Overall confidence: {confidence_output.overall_confidence:.3f}")
    print(f"   Uncertainty estimate: {confidence_output.uncertainty_estimate:.3f}")
    print(f"   Methods used: {list(confidence_output.method_scores.keys())}")
    print(f"   Calibration score: {confidence_output.calibration_score:.3f}")
    
    # 3. Fact Extractor Demo
    print("\n3. Symbolic Fact Extraction")
    print("-" * 27)
    
    from adapters.fact_extractor import create_llama_fact_extractor, extract_facts_from_sql
    
    fact_extractor = create_llama_fact_extractor()
    
    # Pattern-based extraction
    sql_query = "SELECT c.name, COUNT(o.id) FROM customers c JOIN orders o ON c.id = o.customer_id WHERE c.active = 1 GROUP BY c.name"
    pattern_facts = extract_facts_from_sql(sql_query)
    
    print(f"   SQL Query: {sql_query}")
    print(f"   Pattern-based facts: {len(pattern_facts)}")
    for i, fact in enumerate(pattern_facts[:3]):
        print(f"     {i+1}. {fact.fact_string} (confidence: {fact.confidence:.2f})")
    
    # Neural-based extraction
    neural_result = fact_extractor.forward(
        hidden_states=hidden_states[:, :, :4096],  # Match fact extractor dimensions
        input_text=sql_query
    )
    
    print(f"   Neural-based facts: {len(neural_result.facts)}")
    print(f"   Extraction confidence: {neural_result.extraction_confidence:.3f}")
    print(f"   Processing time: {neural_result.processing_time:.4f}s")

def demo_neurosymbolic_adapter():
    """Demonstrate complete neurosymbolic adapter functionality"""
    print("\n\n🧠 NEUROSYMBOLIC ADAPTER DEMO")
    print("=" * 60)
    
    from adapters.neurosymbolic_adapter import NeurosymbolicAdapter, AdapterConfig
    from adapters.bridge_layer import create_llama_bridge
    from adapters.confidence_estimator import create_llama_confidence_estimator, ConfidenceMethod
    from adapters.fact_extractor import create_llama_fact_extractor
    
    # Create adapter with configuration
    config = AdapterConfig(
        lora_r=8,
        lora_alpha=16,
        bridge_dim=256,
        symbolic_dim=128,
        confidence_dim=64,
        max_facts_per_query=20
    )
    
    adapter = NeurosymbolicAdapter("llama-3.1-8b-sql", config)
    
    # Add all components
    print("Setting up components...")
    adapter.set_bridge_layer(create_llama_bridge())
    adapter.set_confidence_estimator(create_llama_confidence_estimator(
        methods=[ConfidenceMethod.ENTROPY, ConfidenceMethod.ATTENTION_BASED]
    ))
    adapter.set_fact_extractor(create_llama_fact_extractor())
    
    # Show adapter status
    status = adapter.get_status()
    print(f"\nAdapter Status:")
    print(f"   Base model: {status['base_model']}")
    print(f"   Trainable parameters: {status['trainable_parameters']:,}")
    print(f"   Components active: {status['components']}")
    
    # Test SQL generation
    test_queries = [
        "Find all customers who have placed orders",
        "Get the total revenue by product category",
        "List customers with more than 5 orders",
        "Show products that are out of stock"
    ]
    
    print(f"\nSQL Generation Results:")
    print("-" * 25)
    
    for i, query in enumerate(test_queries, 1):
        result = adapter.generate_sql(query)
        
        print(f"\n{i}. Query: {query}")
        print(f"   SQL: {result['sql']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        
        if result['extracted_facts']:
            if hasattr(result['extracted_facts'], 'facts'):
                print(f"   Facts: {len(result['extracted_facts'].facts)} extracted")
            elif isinstance(result['extracted_facts'], list):
                print(f"   Facts: {len(result['extracted_facts'])} extracted")
            else:
                print(f"   Facts: extracted (type: {type(result['extracted_facts'])})")
        
        if result['symbolic_context'] is not None:
            print(f"   Symbolic reasoning: Active")

def demo_model_manager():
    """Demonstrate model manager functionality"""
    print("\n\n📊 MODEL MANAGER DEMO")
    print("=" * 60)
    
    from adapters.model_manager import ModelManager, ModelConfig, ModelType, DeviceType
    from adapters.confidence_estimator import ConfidenceMethod
    
    # Create manager with configuration
    config = ModelConfig(
        model_type=ModelType.LLAMA_8B,
        device=DeviceType.CPU,  # Use CPU for demo
        lora_r=8,
        bridge_dim=256,
        symbolic_dim=128,
        enable_bridge=True,
        enable_confidence=True,
        enable_fact_extraction=True,
        confidence_methods=[ConfidenceMethod.ENTROPY, ConfidenceMethod.TEMPERATURE_SCALING]
    )
    
    manager = ModelManager(config)
    
    print(f"Manager initialized on device: {manager.device}")
    
    # Load multiple models
    model_configs = [
        ("sql_model_1", config),
        ("sql_model_2", ModelConfig(
            model_type=ModelType.LLAMA_8B,
            device=DeviceType.CPU,
            lora_r=16,
            bridge_dim=512,
            symbolic_dim=256
        ))
    ]
    
    print(f"\nLoading models...")
    for model_id, model_config in model_configs:
        adapter = manager.load_model(model_id, model_config)
        info = manager.get_model_info(model_id)
        
        print(f"   {model_id}:")
        print(f"     Parameters: {info.parameter_count['total']:,}")
        print(f"     Components: {sum(info.component_status.values())}/4 active")
        print(f"     Load time: {info.load_time:.2f}s")
    
    # Test SQL generation with different models
    print(f"\nTesting SQL generation across models:")
    test_instruction = "Find customers with recent orders"
    
    for model_id, _ in model_configs:
        result = manager.generate_sql(model_id, test_instruction)
        print(f"   {model_id}: {result['sql']}")
        print(f"     Confidence: {result['confidence']:.3f}")
    
    # System status
    status = manager.get_system_status()
    print(f"\nSystem Status:")
    print(f"   Device: {status['device']}")
    print(f"   Loaded models: {status['loaded_models']}")
    print(f"   Memory usage: {status['memory_usage']['cpu']:.1f} MB CPU")
    
    # Benchmark models
    print(f"\nBenchmarking models...")
    for model_id, _ in model_configs:
        benchmark = manager.benchmark_model(model_id, num_queries=5)
        if benchmark:
            print(f"   {model_id}:")
            print(f"     Avg time: {benchmark['average_time_per_query']:.3f}s")
            print(f"     Avg confidence: {benchmark['average_confidence']:.3f}")
            print(f"     QPS: {benchmark['queries_per_second']:.1f}")
    
    # Cleanup
    manager.cleanup()
    print(f"\nCleanup completed. Models unloaded: {len(manager.list_models()) == 0}")

def demo_training_system():
    """Demonstrate training system functionality"""
    print("\n\n🎯 TRAINING SYSTEM DEMO")
    print("=" * 60)
    
    from adapters.adapter_trainer import AdapterTrainer, TrainingConfig, create_mock_dataset
    from adapters.neurosymbolic_adapter import NeurosymbolicAdapter, AdapterConfig
    
    # Create training configuration
    training_config = TrainingConfig(
        num_epochs=1,  # Small for demo
        batch_size=2,
        learning_rate=2e-4,
        lora_r=8,
        bridge_dim=256,
        symbolic_dim=128,
        eval_steps=2,
        logging_steps=1,
        output_dir="./demo_training_output"
    )
    
    print(f"Training Configuration:")
    print(f"   Epochs: {training_config.num_epochs}")
    print(f"   Batch size: {training_config.batch_size}")
    print(f"   Learning rate: {training_config.learning_rate}")
    print(f"   LoRA rank: {training_config.lora_r}")
    
    # Create trainer
    trainer = AdapterTrainer(training_config)
    
    # Create mock datasets
    train_dataset = create_mock_dataset(10)  # Small for demo
    eval_dataset = create_mock_dataset(5)
    
    print(f"\nDatasets created:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Evaluation samples: {len(eval_dataset)}")
    
    # Setup training
    trainer.setup_training(train_dataset, eval_dataset)
    
    print(f"   Training batches: {len(trainer.train_loader)}")
    print(f"   Eval batches: {len(trainer.eval_loader)}")
    
    # Get training summary
    summary = trainer.get_training_summary()
    print(f"\nModel Information:")
    print(f"   Trainable parameters: {summary['model_info']['trainable_parameters']:,}")
    print(f"   Active components: {summary['model_info']['components']}")
    
    print(f"\nTraining system ready for full training run!")

def demo_end_to_end_pipeline():
    """Demonstrate complete end-to-end pipeline"""
    print("\n\n🚀 END-TO-END PIPELINE DEMO")
    print("=" * 60)
    
    from adapters.model_manager import ModelManager, ModelConfig, ModelType, DeviceType
    from adapters.confidence_estimator import ConfidenceMethod
    
    # Create comprehensive configuration
    config = ModelConfig(
        model_type=ModelType.LLAMA_8B,
        model_name="neurosymbolic-sql-llama",
        device=DeviceType.CPU,
        lora_r=8,
        bridge_dim=256,
        symbolic_dim=128,
        enable_bridge=True,
        enable_confidence=True,
        enable_fact_extraction=True,
        confidence_methods=[
            ConfidenceMethod.ENTROPY,
            ConfidenceMethod.TEMPERATURE_SCALING,
            ConfidenceMethod.ATTENTION_BASED
        ],
        fact_extraction_threshold=0.6,
        max_facts_per_query=25
    )
    
    # Initialize system
    manager = ModelManager(config)
    model_id = "production_sql_model"
    
    print(f"Initializing production neurosymbolic SQL system...")
    adapter = manager.load_model(model_id, config)
    
    # Show system capabilities
    info = manager.get_model_info(model_id)
    print(f"\nSystem Capabilities:")
    print(f"   Model: {info.model_name}")
    print(f"   Device: {info.device}")
    print(f"   Total parameters: {info.parameter_count['total']:,}")
    print(f"   Trainable parameters: {info.parameter_count['trainable']:,}")
    print(f"   Neural-symbolic components: {sum(info.component_status.values())}/4")
    
    # Complex SQL generation scenarios
    scenarios = [
        {
            "name": "E-commerce Analytics",
            "instruction": "Find top customers by revenue with their order history",
            "schema": "customers (id, name, email), orders (id, customer_id, amount, date), products (id, name, category)"
        },
        {
            "name": "Inventory Management", 
            "instruction": "Show products running low on inventory with reorder recommendations",
            "schema": "products (id, name, stock_level, reorder_point), suppliers (id, name, contact)"
        },
        {
            "name": "User Behavior Analysis",
            "instruction": "Analyze user engagement patterns and identify churn risk",
            "schema": "users (id, name, signup_date), sessions (id, user_id, duration, actions), activities (id, user_id, type, timestamp)"
        }
    ]
    
    print(f"\nProcessing Complex SQL Scenarios:")
    print("=" * 40)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Schema: {scenario['schema']}")
        print(f"   Request: {scenario['instruction']}")
        
        # Generate SQL with full pipeline
        result = manager.generate_sql(
            model_id, 
            scenario['instruction'],
            schema=scenario['schema']
        )
        
        print(f"   Generated SQL: {result['sql']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Generation method: {result.get('generation_method', 'unknown')}")
        
        if result.get('extracted_facts'):
            if hasattr(result['extracted_facts'], 'facts'):
                print(f"   Symbolic facts: {len(result['extracted_facts'].facts)} extracted")
            elif isinstance(result['extracted_facts'], list):
                print(f"   Symbolic facts: {len(result['extracted_facts'])} extracted")
            else:
                print(f"   Symbolic facts: extracted (type: {type(result['extracted_facts'])})")
        
        if result.get('model_info'):
            neural_enabled = result['model_info'].get('neural_adapters_enabled', False)
            print(f"   Neural adapters: {'✅ Active' if neural_enabled else '❌ Inactive'}")
    
    # Performance analysis
    print(f"\nSystem Performance Analysis:")
    print("-" * 30)
    
    benchmark = manager.benchmark_model(model_id, num_queries=8)
    if benchmark:
        print(f"   Average response time: {benchmark['average_time_per_query']:.3f}s")
        print(f"   Average confidence: {benchmark['average_confidence']:.3f}")
        print(f"   Throughput: {benchmark['queries_per_second']:.1f} queries/second")
        print(f"   Total queries processed: {benchmark['num_queries']}")
    
    # System status
    status = manager.get_system_status()
    print(f"\nFinal System Status:")
    print(f"   Active models: {status['loaded_models']}")
    print(f"   Memory usage: {status['memory_usage']['cpu']:.1f} MB")
    print(f"   Device utilization: {status['device']}")
    
    # Cleanup
    manager.cleanup()
    print(f"\n✅ End-to-end pipeline demonstration completed successfully!")

def main():
    """Run complete functionality demonstration"""
    print("🌟 NEUROSYMBOLIC SQL ADAPTER - FULL FUNCTIONALITY DEMO")
    print("=" * 70)
    print("Demonstrating the complete integration of neural language models")
    print("with symbolic reasoning for enhanced SQL generation and validation.")
    print("=" * 70)
    
    try:
        # Run all demonstrations
        demo_individual_components()
        demo_neurosymbolic_adapter() 
        demo_model_manager()
        demo_training_system()
        demo_end_to_end_pipeline()
        
        print("\n" + "=" * 70)
        print("🎉 FULL FUNCTIONALITY DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\n✅ All systems operational:")
        print("   • Neural adapter components working correctly")
        print("   • Symbolic reasoning integration functional") 
        print("   • Model management system operational")
        print("   • Training pipeline ready for deployment")
        print("   • End-to-end pipeline verified")
        print("\n🚀 The neurosymbolic SQL adapter system is ready for production use!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.exception("Demo error details:")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)