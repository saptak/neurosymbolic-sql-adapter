
# 🚀 Quick Start Guide

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd neurosymbolic-sql-adapter
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv_neurosymbolic
   source venv_neurosymbolic/bin/activate  # On Windows: venv_neurosymbolic\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install pytest tqdm matplotlib psutil pyyaml
   ```

## Basic Usage

### 1. Simple Neural Adapter Usage

```python
from adapters.neurosymbolic_adapter import NeurosymbolicAdapter, AdapterConfig

# Create configuration
config = AdapterConfig(lora_r=8, bridge_dim=256, symbolic_dim=128)

# Initialize adapter
adapter = NeurosymbolicAdapter("llama-3.1-8b-sql", config)

# Generate SQL
result = adapter.generate_sql("Find all customers with recent orders")
print(f"SQL: {result['sql']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### 2. Model Manager Usage

```python
from adapters.model_manager import ModelManager, ModelConfig, ModelType

# Create manager with configuration
config = ModelConfig(model_type=ModelType.LLAMA_8B, device="cpu")
manager = ModelManager(config)

# Load model
adapter = manager.load_model("sql_model", config)

# Generate SQL through manager
result = manager.generate_sql("sql_model", "Get sales by category")
print(f"Generated: {result['sql']}")
```

### 3. Hybrid Model Integration

```python
from integration.hybrid_model import NeurosymbolicSQLModel

# Create hybrid model with neural adapters
model = NeurosymbolicSQLModel(enable_neural_adapters=True)

# Add schema
schema = "customers (id, name, email), orders (id, customer_id, amount)"
model.add_schema(schema)

# Generate and validate SQL
result = model.generate_sql("Find top customers by order value")
print(f"SQL: {result.sql}")
print(f"Valid: {result.is_valid}")
print(f"Method: {result.generation_method}")
```

## Running Tests

```bash
# Run all neural adapter tests
python -m pytest tests/test_neural_adapters.py -v

# Run integration verification
python verify_integration.py

# Run full functionality demo
python demo_full_functionality.py
```

## Next Steps

1. **Explore Documentation**: Check the [API Documentation](../api/index.html) for detailed class and method references
2. **Run Examples**: Try the [example scripts](../examples/index.html) for comprehensive demonstrations  
3. **View PyDoc**: Browse the [PyDoc reference](../pydoc/index.html) for complete API coverage
4. **Check Tests**: Review the test suite for usage patterns and expected behavior

## System Requirements

- **Python**: 3.8+ (3.13+ recommended)
- **PyTorch**: 2.0+ (2.7+ recommended)
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **Storage**: 2GB+ free space

## Troubleshooting

- **Import Errors**: Ensure virtual environment is activated and dependencies installed
- **PyTorch Issues**: Use CPU-only installation for compatibility
- **Memory Issues**: Reduce batch size or model dimensions in configuration
- **Performance**: Enable GPU acceleration if available
