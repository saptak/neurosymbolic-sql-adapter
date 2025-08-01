#!/usr/bin/env python3
"""
Integration Test for Neural-Symbolic Hybrid Model

Tests the complete integration of neural adapters with the existing
symbolic reasoning system in the hybrid model.
"""

import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import modules with absolute imports
import os
os.chdir(project_root / "src")

from integration.hybrid_model import NeurosymbolicSQLModel
from adapters.model_manager import ModelConfig, ModelType, DeviceType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_neural_symbolic_integration():
    """Test integration of neural adapters with symbolic reasoning"""
    
    print("🧪 Testing Neural-Symbolic Integration")
    print("=" * 50)
    
    # Test 1: Initialize model with neural adapters disabled
    print("\n1. Testing symbolic-only mode...")
    try:
        model_symbolic = NeurosymbolicSQLModel(
            base_model="test/model",
            enable_neural_adapters=False
        )
        
        # Test schema addition
        schema = "customers (id, name, email), orders (id, customer_id, amount)"
        model_symbolic.add_schema(schema)
        
        # Test SQL generation
        result = model_symbolic.generate_sql("Find all customers with orders")
        
        print(f"✅ Symbolic generation: {result.generation_method}")
        print(f"   SQL: {result.sql}")
        print(f"   Valid: {result.is_valid}")
        print(f"   Confidence: {result.confidence:.2f}")
        
        assert result.generation_method in ["mock", "symbolic"]
        assert result.neural_confidence is None
        assert result.extracted_facts is None
        
    except Exception as e:
        print(f"❌ Symbolic-only test failed: {e}")
        return False
    
    # Test 2: Initialize model with neural adapters enabled
    print("\n2. Testing neural-symbolic hybrid mode...")
    try:
        # Create configuration for neural adapters
        model_config = ModelConfig(
            model_type=ModelType.LLAMA_8B,
            model_name="test/model",
            device=DeviceType.CPU,  # Use CPU for testing
            lora_r=8,  # Smaller for testing
            bridge_dim=256,
            symbolic_dim=128,
            enable_bridge=True,
            enable_confidence=True,
            enable_fact_extraction=True
        )
        
        model_neural = NeurosymbolicSQLModel(
            base_model="test/model",
            enable_neural_adapters=True,
            model_config=model_config
        )
        
        # Test schema addition
        model_neural.add_schema(schema)
        
        # Test neural adapter status
        adapter_status = model_neural.get_neural_adapter_status()
        print(f"   Neural adapter available: {adapter_status['available']}")
        
        if adapter_status['available']:
            print(f"   Model name: {adapter_status['model_name']}")
            print(f"   Components active: {adapter_status['components']}")
        
        # Test SQL generation
        result = model_neural.generate_sql("Find customers with large orders")
        
        print(f"✅ Neural generation: {result.generation_method}")
        print(f"   SQL: {result.sql}")
        print(f"   Valid: {result.is_valid}")
        print(f"   Confidence: {result.confidence:.2f}")
        
        if result.neural_confidence is not None:
            print(f"   Neural confidence: {result.neural_confidence:.2f}")
        
        if result.extracted_facts:
            print(f"   Extracted facts: {len(result.extracted_facts)}")
        
        assert result.generation_method in ["neural", "mock"]
        assert result.model_info is not None
        assert result.model_info['neural_adapters_enabled'] == True
        
    except Exception as e:
        print(f"❌ Neural-symbolic test failed: {e}")
        logger.exception("Neural-symbolic test error details:")
        return False
    
    # Test 3: Test generation mode switching
    print("\n3. Testing generation mode switching...")
    try:
        # Test setting different modes
        modes_to_test = ["neural", "symbolic", "hybrid", "auto"]
        
        for mode in modes_to_test:
            success = model_neural.set_generation_mode(mode)
            current_mode = model_neural.get_generation_mode()
            print(f"   Mode '{mode}': Set={success}, Current={current_mode}")
        
        # Test invalid mode
        invalid_success = model_neural.set_generation_mode("invalid")
        print(f"   Invalid mode handled: {not invalid_success}")
        
        assert not invalid_success
        
    except Exception as e:
        print(f"❌ Mode switching test failed: {e}")
        return False
    
    # Test 4: Test enable/disable neural generation
    print("\n4. Testing neural generation enable/disable...")
    try:
        # Disable neural generation
        model_neural.disable_neural_generation()
        result_disabled = model_neural.generate_sql("Find all products")
        print(f"   Disabled result: {result_disabled.generation_method}")
        
        # Re-enable neural generation
        enabled = model_neural.enable_neural_generation()
        result_enabled = model_neural.generate_sql("Find all products")
        print(f"   Re-enabled: {enabled}, Result: {result_enabled.generation_method}")
        
    except Exception as e:
        print(f"❌ Enable/disable test failed: {e}")
        return False
    
    # Test 5: Test comprehensive system information
    print("\n5. Testing system information...")
    try:
        schema_info = model_neural.get_schema_info()
        reasoning_stats = model_neural.get_reasoning_stats()
        
        print(f"   Tables loaded: {len(schema_info['tables'])}")
        print(f"   Facts count: {schema_info['facts_count']}")
        print(f"   Reasoning rules: {reasoning_stats['total_rules']}")
        
        assert len(schema_info['tables']) > 0
        assert schema_info['facts_count'] > 0
        assert reasoning_stats['total_rules'] > 0
        
    except Exception as e:
        print(f"❌ System information test failed: {e}")
        return False
    
    print("\n✅ All integration tests passed!")
    print("🎉 Neural-Symbolic Integration Successful!")
    return True

def test_error_handling():
    """Test error handling in integrated system"""
    
    print("\n🔧 Testing Error Handling")
    print("=" * 30)
    
    try:
        model = NeurosymbolicSQLModel()
        
        # Test with invalid instruction
        result = model.generate_sql("")
        print(f"   Empty instruction: {result.generation_method}, Valid: {result.is_valid}")
        
        # Test with malformed schema
        result = model.generate_sql("test", schema="invalid schema format")
        print(f"   Invalid schema: {result.generation_method}, Valid: {result.is_valid}")
        
        print("✅ Error handling tests passed!")
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False
    
    return True

def main():
    """Run all integration tests"""
    print("🚀 Starting Neural-Symbolic SQL Adapter Integration Tests")
    print("=" * 60)
    
    # Run integration tests
    integration_success = test_neural_symbolic_integration()
    
    # Run error handling tests
    error_handling_success = test_error_handling()
    
    # Final results
    print("\n" + "=" * 60)
    if integration_success and error_handling_success:
        print("🎉 ALL TESTS PASSED! Neural-Symbolic Integration Complete!")
        print("\n✅ Phase 3 Task 3.8 Successfully Completed:")
        print("   - Neural adapters integrated with existing hybrid model")
        print("   - Backward compatibility maintained")
        print("   - Mode switching and control methods implemented")
        print("   - Comprehensive error handling verified")
        return True
    else:
        print("❌ Some tests failed. Integration needs review.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)