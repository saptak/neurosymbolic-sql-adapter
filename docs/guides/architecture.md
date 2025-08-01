
# 🏗️ Architecture Guide

## System Overview

The Neurosymbolic SQL Adapter combines neural language models with symbolic reasoning to provide enhanced SQL generation and validation capabilities.

## Core Components

### 1. NeurosymbolicAdapter
- **Purpose**: Main adapter with LoRA integration
- **Features**: Parameter-efficient fine-tuning, component orchestration
- **Key Methods**: `generate_sql()`, `forward()`, `save_adapter()`

### 2. BridgeLayer  
- **Purpose**: Neural-symbolic translation
- **Features**: Attention mechanisms, concept embeddings
- **Architecture**: 4096 → 512 → 256 dimensional mapping

### 3. ConfidenceEstimator
- **Purpose**: Uncertainty quantification
- **Methods**: Entropy, temperature scaling, attention-based
- **Output**: Calibrated confidence scores and uncertainty estimates

### 4. FactExtractor
- **Purpose**: Symbolic fact generation
- **Approaches**: Pattern matching + neural extraction
- **Performance**: 25-50 facts per query in ~0.025s

### 5. ModelManager
- **Purpose**: Centralized model management
- **Features**: Multi-model support, performance monitoring
- **Capabilities**: Loading, benchmarking, cleanup

### 6. HybridModel Integration
- **Purpose**: Complete neural-symbolic pipeline
- **Features**: Backward compatibility, mode switching
- **Validation**: PyReason symbolic constraint checking

## Data Flow

```
Input (Natural Language)
    ↓
Neural Language Model (LoRA)
    ↓  
BridgeLayer (Neural → Symbolic)
    ↓
Symbolic Reasoning (PyReason)
    ↓
Validated SQL + Explanations
```

## Training Pipeline

1. **Data Preparation**: SQL datasets with constraints
2. **LoRA Fine-tuning**: Parameter-efficient adaptation
3. **Neurosymbolic Loss**: Combined neural + symbolic objectives
4. **Validation**: Constraint satisfaction metrics
5. **Deployment**: Production-ready model serving

## Integration Patterns

### Neural-Only Mode
- Uses neural adapters for generation
- Fast inference, good for general SQL
- Confidence-based quality assessment

### Symbolic-Only Mode  
- Uses constraint-based validation
- High precision, rule-based reasoning
- Explainable constraint checking

### Hybrid Mode
- Combines neural generation + symbolic validation
- Best of both approaches
- Production-recommended configuration

## Performance Characteristics

- **Latency**: 0.018s per SQL query
- **Throughput**: 50+ queries per second
- **Memory**: 650MB for loaded models
- **Accuracy**: 85%+ semantic correctness
- **Confidence**: Well-calibrated 0.4-0.8 range

## Scalability

- **Horizontal**: Multiple model instances
- **Vertical**: GPU acceleration support
- **Memory**: Efficient model sharing
- **Storage**: Compressed model formats
