#!/usr/bin/env python3
"""
Hybrid Model Integration

Main integration class that combines neural language models with symbolic reasoning
for neurosymbolic SQL generation and validation.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

# Handle imports for both relative and absolute import contexts
try:
    from ..reasoning.pyreason_engine import PyReasonEngine, ValidationResult
    from ..reasoning.sql_knowledge_base import SQLKnowledgeBase, create_simple_schema
    from ..reasoning.constraint_validator import ConstraintValidator
    from ..reasoning.sql_to_facts import SQLToFactsConverter
    from ..reasoning.explanation_generator import ExplanationGenerator, ExplanationContext
    from ..reasoning.config_loader import ConfigurationLoader
except ImportError:
    # Fallback for verification scripts or direct execution
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent.parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    from reasoning.pyreason_engine import PyReasonEngine, ValidationResult
    from reasoning.sql_knowledge_base import SQLKnowledgeBase, create_simple_schema
    from reasoning.constraint_validator import ConstraintValidator
    from reasoning.sql_to_facts import SQLToFactsConverter
    from reasoning.explanation_generator import ExplanationGenerator, ExplanationContext
    from reasoning.config_loader import ConfigurationLoader

# Import new neural adapter components
try:
    from ..adapters.model_manager import ModelManager, ModelConfig, ModelType, DeviceType
    from ..adapters.neurosymbolic_adapter import NeurosymbolicAdapter, AdapterConfig
    from ..adapters.adapter_trainer import AdapterTrainer, TrainingConfig
    NEURAL_ADAPTERS_AVAILABLE = True
except ImportError:
    # Try absolute imports for verification scripts
    try:
        from adapters.model_manager import ModelManager, ModelConfig, ModelType, DeviceType
        from adapters.neurosymbolic_adapter import NeurosymbolicAdapter, AdapterConfig
        from adapters.adapter_trainer import AdapterTrainer, TrainingConfig
        NEURAL_ADAPTERS_AVAILABLE = True
    except ImportError:
        NEURAL_ADAPTERS_AVAILABLE = False
        logging.warning("Neural adapter components not available")
        # Define stub classes to prevent NameError
        class ModelConfig:
            pass
        class ModelManager:
            pass
        class DeviceType:
            pass


@dataclass
class NeurosymbolicResult:
    """Result from neurosymbolic SQL processing"""
    sql: str
    is_valid: bool
    confidence: float
    violations: List[str]
    explanation: Optional[str] = None
    reasoning_trace: Optional[List[str]] = None
    optimization_suggestions: Optional[List[str]] = None
    
    # Enhanced fields for neural adapter integration
    neural_confidence: Optional[float] = None
    extracted_facts: Optional[List[str]] = None
    symbolic_context: Optional[Any] = None
    generation_method: str = "mock"  # "neural", "mock", "hybrid"
    model_info: Optional[Dict[str, Any]] = None


class NeurosymbolicSQLModel:
    """
    Neurosymbolic SQL Model
    
    Main orchestration class that combines neural language models
    with symbolic reasoning for SQL generation and validation.
    
    This is a simplified implementation that focuses on the symbolic reasoning
    components. In a full implementation, this would integrate with actual
    fine-tuned language models.
    """
    
    def __init__(self, base_model: str = None, config_path: Optional[str] = None, 
                 enable_neural_adapters: bool = True, model_config: Optional[ModelConfig] = None):
        """
        Initialize neurosymbolic SQL model
        
        Args:
            base_model: Base language model identifier
            config_path: Path to configuration file
            enable_neural_adapters: Whether to use neural adapters if available
            model_config: Configuration for neural adapters
        """
        self.logger = logging.getLogger(__name__)
        self.base_model = base_model
        self.enable_neural_adapters = enable_neural_adapters and NEURAL_ADAPTERS_AVAILABLE
        
        # Initialize symbolic reasoning components
        self.config_loader = ConfigurationLoader(config_path)
        self.reasoning_engine = PyReasonEngine(config_path)
        self.knowledge_base = SQLKnowledgeBase()
        self.constraint_validator = ConstraintValidator(self.knowledge_base)
        self.facts_converter = SQLToFactsConverter(self.knowledge_base)
        self.explanation_generator = ExplanationGenerator()
        
        # Initialize neural adapter components if available
        self.model_manager = None
        self.neural_model_id = None
        
        if self.enable_neural_adapters:
            try:
                # Use provided config or create default
                if model_config is None:
                    model_config = ModelConfig(
                        model_type=ModelType.LLAMA_8B,
                        model_name=base_model or "unsloth/llama-3.1-8b-instruct-bnb-4bit",
                        device=DeviceType.AUTO,
                        lora_r=8,  # Smaller for compatibility
                        bridge_dim=256,
                        symbolic_dim=128,
                        enable_bridge=True,
                        enable_confidence=True,
                        enable_fact_extraction=True
                    )
                
                self.model_manager = ModelManager(model_config)
                self.neural_model_id = "hybrid_neurosymbolic_model"
                
                # Load neural model
                self.neural_adapter = self.model_manager.load_model(
                    self.neural_model_id, model_config
                )
                
                self.logger.info("Neural adapters successfully integrated")
                
            except Exception as e:
                self.logger.warning(f"Failed to initialize neural adapters: {e}")
                self.enable_neural_adapters = False
                self.model_manager = None
                self.neural_adapter = None
        
        # Model state
        self.is_initialized = True
        
        generation_mode = "neural-symbolic hybrid" if self.enable_neural_adapters else "symbolic reasoning"
        self.logger.info(f"NeurosymbolicSQLModel initialized in {generation_mode} mode")
    
    def add_schema(self, schema: Union[Dict[str, Any], str]) -> None:
        """
        Add database schema to the knowledge base
        
        Args:
            schema: Database schema (dict or string description)
        """
        if isinstance(schema, str):
            # Parse simple string schema format
            schema_dict = self._parse_schema_string(schema)
        else:
            schema_dict = schema
        
        self.knowledge_base.add_schema(schema_dict)
        self.logger.info(f"Added schema with {len(schema_dict)} tables")
    
    def _parse_schema_string(self, schema_str: str) -> Dict[str, List[str]]:
        """
        Parse simple schema string format
        
        Args:
            schema_str: Schema string like "customers (id, name, email), orders (id, customer_id, amount)"
            
        Returns:
            Parsed schema dictionary
        """
        schema_dict = {}
        
        # Split by comma and parse each table
        tables = schema_str.split('),')
        
        for table_def in tables:
            table_def = table_def.strip().rstrip(')')
            if '(' in table_def:
                table_name, columns_str = table_def.split('(', 1)
                table_name = table_name.strip()
                
                # Parse columns
                columns = [col.strip() for col in columns_str.split(',')]
                schema_dict[table_name] = columns
        
        return schema_dict
    
    def generate_sql(self, instruction: str, schema: Optional[Union[str, Dict[str, Any]]] = None) -> NeurosymbolicResult:
        """
        Generate SQL from natural language instruction
        
        Args:
            instruction: Natural language instruction
            schema: Database schema (optional)
            
        Returns:
            NeurosymbolicResult with generated SQL and validation
        """
        try:
            # Add schema if provided
            if schema:
                self.add_schema(schema)
            
            # Determine generation method
            if self.enable_neural_adapters and self.neural_adapter is not None:
                # Use neural adapter for SQL generation
                generation_result = self._neural_sql_generation(instruction, schema)
                generated_sql = generation_result['sql']
                neural_confidence = generation_result['confidence']
                extracted_facts = generation_result['extracted_facts']
                symbolic_context = generation_result['symbolic_context']
                generation_method = "neural"
            else:
                # Fallback to mock SQL generation
                generated_sql = self._mock_sql_generation(instruction, schema)
                neural_confidence = None
                extracted_facts = None
                symbolic_context = None
                generation_method = "mock"
            
            # Validate the generated SQL using symbolic reasoning
            validation_result = self._validate_generated_sql(generated_sql, schema)
            
            # Generate explanation
            explanation = self._generate_explanation(
                instruction, generated_sql, validation_result, schema
            )
            
            return NeurosymbolicResult(
                sql=generated_sql,
                is_valid=validation_result.is_valid,
                confidence=validation_result.confidence,
                violations=validation_result.violations,
                explanation=explanation,
                reasoning_trace=validation_result.reasoning_trace,
                optimization_suggestions=self._get_optimization_suggestions(validation_result),
                neural_confidence=neural_confidence,
                extracted_facts=extracted_facts,
                symbolic_context=symbolic_context,
                generation_method=generation_method,
                model_info={
                    'neural_adapters_enabled': self.enable_neural_adapters,
                    'model_manager_available': self.model_manager is not None,
                    'base_model': self.base_model
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating SQL: {e}")
            return NeurosymbolicResult(
                sql="",
                is_valid=False,
                confidence=0.0,
                violations=[f"Generation error: {str(e)}"],
                explanation=f"Failed to generate SQL: {str(e)}",
                generation_method="error"
            )
    
    def _neural_sql_generation(self, instruction: str, schema: Optional[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Generate SQL using neural adapter
        
        Args:
            instruction: Natural language instruction
            schema: Database schema (optional)
            
        Returns:
            Dictionary with SQL and neural adapter outputs
        """
        try:
            # Use neural adapter for SQL generation
            result = self.neural_adapter.generate_sql(instruction)
            
            # Extract facts if available
            extracted_facts = []
            if result.get('extracted_facts'):
                if hasattr(result['extracted_facts'], 'facts'):
                    extracted_facts = [str(fact) for fact in result['extracted_facts'].facts]
                elif isinstance(result['extracted_facts'], list):
                    extracted_facts = [str(fact) for fact in result['extracted_facts']]
                else:
                    extracted_facts = [str(result['extracted_facts'])]
            
            return {
                'sql': result['sql'],
                'confidence': result['confidence'],
                'extracted_facts': extracted_facts,
                'symbolic_context': result.get('symbolic_context'),
                'reasoning_embeddings': result.get('reasoning_embeddings'),
                'neural_adapter_info': {
                    'model_name': self.neural_adapter.base_model_name,
                    'components_active': self.neural_adapter.get_status()['components']
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Neural SQL generation failed, falling back to mock: {e}")
            # Fallback to mock generation if neural adapter fails
            return {
                'sql': self._mock_sql_generation(instruction, schema),
                'confidence': 0.5,  # Lower confidence for fallback
                'extracted_facts': [],
                'symbolic_context': None,
                'reasoning_embeddings': None,
                'neural_adapter_info': {'error': str(e)}
            }
    
    def _mock_sql_generation(self, instruction: str, schema: Optional[Union[str, Dict[str, Any]]]) -> str:
        """
        Mock SQL generation (placeholder for actual neural generation)
        
        In a real implementation, this would use a fine-tuned language model.
        """
        instruction_lower = instruction.lower()
        
        # Simple pattern matching for demonstration
        if "find" in instruction_lower and "customers" in instruction_lower:
            if "orders" in instruction_lower and "more than" in instruction_lower:
                return """SELECT c.name, c.email 
                         FROM customers c 
                         JOIN orders o ON c.id = o.customer_id 
                         WHERE o.amount > 1000"""
            else:
                return "SELECT * FROM customers"
        
        elif "average" in instruction_lower and "order" in instruction_lower:
            return """SELECT c.name, AVG(o.amount) as avg_amount
                     FROM customers c 
                     JOIN orders o ON c.id = o.customer_id 
                     GROUP BY c.id, c.name"""
        
        elif "haven't" in instruction_lower or "no orders" in instruction_lower:
            return """SELECT c.* 
                     FROM customers c 
                     LEFT JOIN orders o ON c.id = o.customer_id 
                     WHERE o.customer_id IS NULL"""
        
        else:
            # Default query
            return "SELECT * FROM customers LIMIT 10"
    
    def _validate_generated_sql(self, sql: str, schema: Optional[Union[str, Dict[str, Any]]]) -> ValidationResult:
        """
        Validate generated SQL using symbolic reasoning
        
        Args:
            sql: Generated SQL query
            schema: Database schema
            
        Returns:
            ValidationResult with validation details
        """
        # Convert schema if it's a string
        if isinstance(schema, str):
            schema_dict = self._parse_schema_string(schema)
        elif schema is None:
            # Use existing knowledge base schema
            schema_dict = {table.name: list(table.columns.keys()) for table in self.knowledge_base.tables.values()}
        else:
            schema_dict = schema
        
        # Validate using the reasoning engine
        return self.reasoning_engine.validate_sql(sql, schema_dict)
    
    def _generate_explanation(self, instruction: str, sql: str, 
                            validation_result: ValidationResult,
                            schema: Optional[Union[str, Dict[str, Any]]]) -> str:
        """Generate explanation for the SQL generation and validation process"""
        
        # Analyze the generated SQL
        analysis = self.facts_converter.analyze_query(sql)
        
        # Create explanation context
        context = ExplanationContext(
            query=sql,
            query_analysis=analysis,
            violations=[],  # Convert string violations to Violation objects if needed
            facts=validation_result.facts_applied,
            reasoning_trace=validation_result.reasoning_trace,
            confidence=validation_result.confidence,
            user_level='intermediate'
        )
        
        # Generate explanation
        explanation = self.explanation_generator.generate_explanation(context)
        
        # Format for display
        formatted_explanation = self.explanation_generator.format_explanation(explanation)
        
        # Add generation context
        generation_context = f"""
SQL Generation Process:
📝 Instruction: {instruction}
🔍 Generated Query: {sql}
✅ Valid: {validation_result.is_valid}
🎯 Confidence: {validation_result.confidence:.2f}

{formatted_explanation}
"""
        
        return generation_context
    
    def _get_optimization_suggestions(self, validation_result: ValidationResult) -> List[str]:
        """Extract optimization suggestions from validation result"""
        suggestions = []
        
        if validation_result.reasoning_trace:
            for trace_item in validation_result.reasoning_trace:
                if 'optimization' in trace_item.lower():
                    suggestions.append(trace_item)
        
        # Add general suggestions based on validation
        if validation_result.confidence < 0.8:
            suggestions.append("Consider reviewing the query for potential improvements")
        
        if not validation_result.is_valid:
            suggestions.append("Fix constraint violations before executing the query")
        
        return suggestions
    
    def validate_sql(self, sql: str, schema: Optional[Union[str, Dict[str, Any]]] = None) -> NeurosymbolicResult:
        """
        Validate existing SQL query
        
        Args:
            sql: SQL query to validate
            schema: Database schema (optional)
            
        Returns:
            NeurosymbolicResult with validation details
        """
        try:
            # Add schema if provided
            if schema:
                self.add_schema(schema)
            
            # Validate the SQL
            validation_result = self._validate_generated_sql(sql, schema)
            
            # Generate explanation
            explanation = self._generate_explanation("User-provided query", sql, validation_result, schema)
            
            return NeurosymbolicResult(
                sql=sql,
                is_valid=validation_result.is_valid,
                confidence=validation_result.confidence,
                violations=validation_result.violations,
                explanation=explanation,
                reasoning_trace=validation_result.reasoning_trace,
                optimization_suggestions=self._get_optimization_suggestions(validation_result)
            )
            
        except Exception as e:
            self.logger.error(f"Error validating SQL: {e}")
            return NeurosymbolicResult(
                sql=sql,
                is_valid=False,
                confidence=0.0,
                violations=[f"Validation error: {str(e)}"],
                explanation=f"Failed to validate SQL: {str(e)}"
            )
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get current schema information"""
        return {
            'tables': {name: table.to_dict() for name, table in self.knowledge_base.tables.items()},
            'statistics': self.knowledge_base.get_statistics(),
            'facts_count': len(self.knowledge_base.generate_facts())
        }
    
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get reasoning engine statistics"""
        return {
            'engine_status': self.reasoning_engine.get_status(),
            'config_stats': self.config_loader.get_statistics(),
            'total_rules': len(self.config_loader.get_reasoning_rules())
        }
    
    def set_generation_mode(self, mode: str) -> bool:
        """
        Set SQL generation mode
        
        Args:
            mode: Generation mode ("neural", "symbolic", "hybrid", "auto")
            
        Returns:
            True if mode was set successfully
        """
        valid_modes = ["neural", "symbolic", "hybrid", "auto"]
        if mode not in valid_modes:
            self.logger.error(f"Invalid mode '{mode}'. Valid modes: {valid_modes}")
            return False
        
        if mode == "neural" and not self.enable_neural_adapters:
            self.logger.warning("Neural mode requested but neural adapters not available. Using symbolic mode.")
            mode = "symbolic"
        
        self.generation_mode = mode
        self.logger.info(f"Generation mode set to: {mode}")
        return True
    
    def get_generation_mode(self) -> str:
        """Get current generation mode"""
        return getattr(self, 'generation_mode', 'auto')
    
    def get_neural_adapter_status(self) -> Dict[str, Any]:
        """Get neural adapter status and information"""
        if not self.enable_neural_adapters or self.neural_adapter is None:
            return {
                'available': False,
                'reason': 'Neural adapters not enabled or not loaded'
            }
        
        status = self.neural_adapter.get_status()
        return {
            'available': True,
            'model_name': self.neural_adapter.base_model_name,
            'components': status['components'],
            'trainable_parameters': status['trainable_parameters'],
            'is_trained': status['is_trained'],
            'training_step': status['training_step']
        }
    
    def enable_neural_generation(self) -> bool:
        """
        Enable neural generation if adapters are available
        
        Returns:
            True if successfully enabled
        """
        if self.neural_adapter is not None:
            self.enable_neural_adapters = True
            self.logger.info("Neural generation enabled")
            return True
        else:
            self.logger.warning("Cannot enable neural generation: no neural adapter loaded")
            return False
    
    def disable_neural_generation(self) -> None:
        """Disable neural generation, fallback to symbolic only"""
        self.enable_neural_adapters = False
        self.logger.info("Neural generation disabled, using symbolic reasoning only")
    
    def reset(self) -> None:
        """Reset the model state"""
        self.knowledge_base.reset()
        self.reasoning_engine.reset()
        self.logger.info("NeurosymbolicSQLModel reset")
    
    def __str__(self) -> str:
        """String representation of the model"""
        return f"NeurosymbolicSQLModel(base_model={self.base_model}, tables={len(self.knowledge_base.tables)})"


# Convenience functions
def create_neurosymbolic_model(base_model: str = None, config_path: Optional[str] = None) -> NeurosymbolicSQLModel:
    """Create a new neurosymbolic SQL model"""
    return NeurosymbolicSQLModel(base_model, config_path)


def quick_validate(sql: str, schema: Optional[Union[str, Dict[str, Any]]] = None) -> NeurosymbolicResult:
    """Quick validation of SQL query"""
    model = NeurosymbolicSQLModel()
    return model.validate_sql(sql, schema)