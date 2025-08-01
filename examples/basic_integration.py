#!/usr/bin/env python3
"""
Basic Integration Example

Demonstrates basic usage of the Neurosymbolic SQL Adapter.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.integration.hybrid_model import NeurosymbolicSQLModel
from src.reasoning.pyreason_engine import PyReasonEngine

def main():
    print("🚀 Neurosymbolic SQL Adapter - Basic Integration Example")
    
    # Initialize the hybrid model
    print("\n1. Initializing Neurosymbolic SQL Model...")
    try:
        model = NeurosymbolicSQLModel(
            base_model="unsloth/llama-3.1-8b-instruct-bnb-4bit"
        )
        print("✅ Model initialized successfully")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return
    
    # Example SQL generation and validation
    examples = [
        {
            "instruction": "Find all customers who have placed orders worth more than $1000",
            "schema": "customers (id, name, email), orders (id, customer_id, amount, order_date)"
        },
        {
            "instruction": "Get the average order amount per customer",
            "schema": "customers (id, name, email), orders (id, customer_id, amount, order_date)"
        },
        {
            "instruction": "Find customers who haven't placed any orders",
            "schema": "customers (id, name, email), orders (id, customer_id, amount, order_date)"
        }
    ]
    
    print("\n2. Generating and Validating SQL Queries...")
    
    for i, example in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"Instruction: {example['instruction']}")
        print(f"Schema: {example['schema']}")
        
        try:
            # Generate SQL with neurosymbolic validation
            result = model.generate_sql(
                instruction=example['instruction'],
                schema=example['schema']
            )
            
            print(f"\n🔮 Generated SQL:")
            print(f"```sql\n{result.sql}\n```")
            
            print(f"\n✅ Validation Results:")
            print(f"  Valid: {result.is_valid}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Reasoning Steps: {len(result.reasoning_trace)}")
            
            if result.explanation:
                print(f"\n💡 Explanation:")
                print(f"  {result.explanation}")
            
            if not result.is_valid and result.violations:
                print(f"\n⚠️  Constraint Violations:")
                for violation in result.violations:
                    print(f"  - {violation}")
                    
        except Exception as e:
            print(f"❌ Error processing example: {e}")
    
    print("\n3. Testing Symbolic Reasoning Engine...")
    try:
        # Test PyReason engine directly
        reasoning_engine = PyReasonEngine()
        
        # Test constraint checking
        test_facts = [
            "primary_key(id, customers)",
            "foreign_key(customer_id, orders, customers)",
            "not_null(name, customers)"
        ]
        
        violations = reasoning_engine.check_constraints(test_facts)
        print(f"✅ Constraint checking completed: {len(violations)} violations found")
        
    except Exception as e:
        print(f"❌ Reasoning engine test failed: {e}")
    
    print("\n🎉 Basic integration example completed!")
    print("\nNext steps:")
    print("- Try advanced_reasoning.py for more complex examples")
    print("- Explore the evaluation framework")
    print("- Customize the reasoning rules in configs/")

if __name__ == "__main__":
    main()