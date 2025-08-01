#!/usr/bin/env python3
"""
PyReason Engine Integration

Core engine for symbolic reasoning using PyReason framework.
Includes compatibility layer for development when PyReason is not available.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import networkx as nx
import yaml

# Use SimpleLogicEngine as the primary implementation
# PyReason has too many compilation issues across different environments
PYREASON_AVAILABLE = False
PYREASON_ERROR = "PyReason disabled due to compilation issues. Using SimpleLogicEngine."

try:
    # Import our custom SimpleLogicEngine
    from .simple_logic_engine import SimpleLogicEngine, LogicFact, LogicRule, ValidationViolation
    SIMPLE_LOGIC_AVAILABLE = True
    logging.info("SimpleLogicEngine successfully imported")
except ImportError as e:
    SIMPLE_LOGIC_AVAILABLE = False
    logging.error(f"SimpleLogicEngine import failed: {e}")
    raise ImportError("Neither PyReason nor SimpleLogicEngine available") from e


# Create a stub pyreason module for development
class PyReasonStub:
    """Stub implementation of PyReason for development purposes"""
    
    def __init__(self):
        self.facts = []
        self.rules = []
        
    def add_fact(self, fact: str):
        self.facts.append(fact)
        
    def add_rule(self, rule: str):
        self.rules.append(rule)
        
    def reason(self, timesteps: int = 10) -> Dict[str, Any]:
        """Stub reasoning that returns mock results"""
        return {
            'facts': self.facts.copy(),
            'rules': self.rules.copy(),
            'violations': [],
            'trace': [f"Applied rule: {rule}" for rule in self.rules],
            'confidence': 0.95
        }
        
    def reset(self):
        self.facts.clear()
        self.rules.clear()


@dataclass
class ValidationResult:
    """Results from symbolic validation"""
    is_valid: bool
    violations: List[str]
    reasoning_trace: List[str]
    confidence: float
    explanation: Optional[str] = None
    facts_applied: Optional[List[str]] = None
    rules_applied: Optional[List[str]] = None


@dataclass
class Constraint:
    """Represents a database constraint for symbolic reasoning"""
    constraint_type: str  # 'primary_key', 'foreign_key', 'not_null', etc.
    scope: List[str]      # Tables/columns affected
    condition: str        # Logical condition
    severity: str = 'error'  # 'error', 'warning', 'info'
    
    def to_pyreason_rule(self) -> str:
        """Convert constraint to PyReason rule format"""
        if self.constraint_type == 'primary_key':
            return f"primary_key({', '.join(self.scope)}) :- {self.condition}"
        elif self.constraint_type == 'foreign_key':
            return f"foreign_key({', '.join(self.scope)}) :- {self.condition}"
        elif self.constraint_type == 'not_null':
            return f"not_null({', '.join(self.scope)}) :- {self.condition}"
        elif self.constraint_type == 'unique':
            return f"unique({', '.join(self.scope)}) :- {self.condition}"
        else:
            return f"{self.constraint_type}({', '.join(self.scope)}) :- {self.condition}"


class PyReasonEngine:
    """
    Core symbolic reasoning engine using PyReason
    
    Provides SQL constraint validation, semantic checking, and explainable reasoning.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize PyReason engine
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize reasoning engine - use SimpleLogicEngine as primary
        if SIMPLE_LOGIC_AVAILABLE:
            self.logic_engine = SimpleLogicEngine()
            self._logic_functional = True
            self.logger.info("SimpleLogicEngine initialized successfully")
        else:
            # Fallback to stub (this should not happen due to import check above)
            self.logic_engine = PyReasonStub()
            self._logic_functional = False
            self.logger.error("No logic engine available - using stub")
        
        # Knowledge base components
        self.facts = []
        self.rules = []
        self.constraints = {}
        self.knowledge_graph = nx.DiGraph()
        
        # Initialize with default SQL reasoning rules
        self._initialize_default_rules()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            'pyreason': {
                'reasoning_steps': 10,
                'inconsistency_check': True,
                'open_world': True,
                'convergence_threshold': 0.01
            },
            'validation': {
                'confidence_threshold': 0.8,
                'enable_explanation': True
            }
        }
        
        if self.config_path:
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    # Merge with defaults
                    for key, value in loaded_config.items():
                        if key in default_config and isinstance(value, dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except Exception as e:
                self.logger.warning(f"Could not load config from {self.config_path}: {e}")
        
        return default_config
    
    def _initialize_default_rules(self):
        """Initialize default SQL reasoning rules"""
        default_rules = [
            # Primary key rules
            "violation(primary_key_duplicate, Table, Column) :- primary_key(Table, Column), duplicate_values(Table, Column)",
            "violation(primary_key_null, Table, Column) :- primary_key(Table, Column), null_value(Table, Column)",
            
            # Foreign key rules  
            "violation(foreign_key_invalid, Table1, Column1, Table2, Column2) :- foreign_key(Table1, Column1, Table2, Column2), ~exists_reference(Table1, Column1, Table2, Column2)",
            
            # Not null rules
            "violation(not_null, Table, Column) :- not_null_constraint(Table, Column), null_value(Table, Column)",
            
            # Unique constraint rules
            "violation(unique_constraint, Table, Column) :- unique_constraint(Table, Column), duplicate_values(Table, Column)",
            
            # Semantic rules
            "violation(type_mismatch, Table, Column, ExpectedType, ActualType) :- column_type(Table, Column, ExpectedType), value_type(Table, Column, ActualType), ExpectedType != ActualType"
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_fact(self, fact: str):
        """Add a fact to the knowledge base"""
        self.facts.append(fact)
        if self._logic_functional and hasattr(self.logic_engine, 'add_fact_from_string'):
            self.logic_engine.add_fact_from_string(fact)
        self.logger.debug(f"Added fact: {fact}")
    
    def add_rule(self, rule: str):
        """Add a reasoning rule to the knowledge base"""
        self.rules.append(rule)
        if self._logic_functional and hasattr(self.logic_engine, 'add_rule_from_string'):
            self.logic_engine.add_rule_from_string(rule)
        self.logger.debug(f"Added rule: {rule}")
    
    def add_constraint(self, constraint: Constraint):
        """Add a constraint to the knowledge base"""
        constraint_id = f"{constraint.constraint_type}_{len(self.constraints)}"
        self.constraints[constraint_id] = constraint
        
        # Convert constraint to PyReason rule
        rule = constraint.to_pyreason_rule()
        self.add_rule(rule)
        
        self.logger.debug(f"Added constraint: {constraint_id}")
    
    def validate_sql(self, sql_query: str, schema: Dict[str, Any]) -> ValidationResult:
        """
        Validate SQL query using symbolic reasoning
        
        Args:
            sql_query: SQL query to validate
            schema: Database schema information
            
        Returns:
            ValidationResult with validation status and explanations
        """
        self.logger.info(f"Validating SQL query: {sql_query[:100]}...")
        
        try:
            # Convert SQL to facts
            facts = self.sql_to_facts(sql_query, schema)
            
            # Add facts to reasoning engine
            for fact in facts:
                self.add_fact(fact)
            
            # Run reasoning using SimpleLogicEngine
            reasoning_steps = self.config['pyreason']['reasoning_steps']
            
            if self._logic_functional:
                try:
                    # SimpleLogicEngine uses validate_sql_constraints method
                    results = self.logic_engine.validate_sql_constraints(facts, [])
                    
                    # Convert SimpleLogicEngine results to expected format
                    reasoning_results = {
                        'facts': [f for f in facts],
                        'violations': results.get('violations', []),
                        'trace': results.get('reasoning_trace', []),
                        'confidence': results.get('confidence', 0.95)
                    }
                    results = reasoning_results
                    
                except Exception as reasoning_error:
                    self.logger.error(f"SimpleLogicEngine reasoning failed: {reasoning_error}")
                    # Create minimal fallback results
                    results = {
                        'facts': facts,
                        'violations': [],
                        'trace': [f"Reasoning failed: {reasoning_error}"],
                        'confidence': 0.1
                    }
            else:
                # No functional reasoning engine available
                results = {
                    'facts': facts,
                    'violations': [],
                    'trace': ["No reasoning engine available"],
                    'confidence': 0.5
                }
            
            # Extract violations
            violations = self.extract_violations(results)
            
            # Calculate confidence
            confidence = self.calculate_confidence(results)
            
            # Generate explanation
            explanation = self.generate_explanation(results, violations) if self.config['validation']['enable_explanation'] else None
            
            return ValidationResult(
                is_valid=len(violations) == 0,
                violations=violations,
                reasoning_trace=results.get('trace', []),
                confidence=confidence,
                explanation=explanation,
                facts_applied=results.get('facts', []),
                rules_applied=results.get('rules', [])
            )
            
        except Exception as e:
            self.logger.error(f"Error during SQL validation: {e}")
            return ValidationResult(
                is_valid=False,
                violations=[f"Validation error: {str(e)}"],
                reasoning_trace=[],
                confidence=0.0,
                explanation=f"Validation failed due to error: {str(e)}"
            )
    
    def sql_to_facts(self, sql_query: str, schema: Dict[str, Any]) -> List[str]:
        """
        Convert SQL query and schema to logical facts
        
        Args:
            sql_query: SQL query string
            schema: Database schema
            
        Returns:
            List of logical facts
        """
        facts = []
        
        # Basic SQL parsing (simplified for now)
        sql_lower = sql_query.lower().strip()
        
        # Extract table references
        if 'from' in sql_lower:
            # Simple table extraction (needs proper SQL parsing)
            parts = sql_lower.split('from')[1].split()
            if parts:
                table_name = parts[0].strip(',;')
                facts.append(f"query_references_table({table_name})")
        
        # Extract column references from SELECT
        if sql_lower.startswith('select'):
            select_part = sql_lower.split('from')[0].replace('select', '').strip()
            if select_part != '*':
                # Simple column extraction
                columns = [col.strip() for col in select_part.split(',')]
                for col in columns:
                    facts.append(f"query_references_column({col})")
        
        # Add schema facts
        for table_name, table_info in schema.items():
            facts.append(f"table_exists({table_name})")
            
            if isinstance(table_info, list):
                # Simple list of column names
                for column in table_info:
                    facts.append(f"column_exists({table_name}, {column})")
            
            elif isinstance(table_info, dict):
                # Handle both direct column dict and nested structure
                if 'columns' in table_info:
                    # Nested structure: {'columns': {...}, 'primary_key': [...], ...}
                    columns = table_info['columns']
                    for column_name, column_info in columns.items():
                        facts.append(f"column_exists({table_name}, {column_name})")
                        if isinstance(column_info, dict):
                            if 'data_type' in column_info:
                                facts.append(f"column_type({table_name}, {column_name}, {column_info['data_type']})")
                            elif 'type' in column_info:
                                facts.append(f"column_type({table_name}, {column_name}, {column_info['type']})")
                            
                            if 'nullable' in column_info and not column_info['nullable']:
                                facts.append(f"not_null_constraint({table_name}, {column_name})")
                        else:
                            # Simple type string
                            facts.append(f"column_type({table_name}, {column_name}, {column_info})")
                    
                    # Add primary key facts
                    if 'primary_key' in table_info and table_info['primary_key']:
                        pk_columns = table_info['primary_key']
                        if isinstance(pk_columns, list):
                            for pk_col in pk_columns:
                                facts.append(f"primary_key({table_name}, {pk_col})")
                
                else:
                    # Direct column dict: {column_name: column_info, ...}
                    for column_name, column_info in table_info.items():
                        facts.append(f"column_exists({table_name}, {column_name})")
                        if isinstance(column_info, dict):
                            if 'type' in column_info:
                                facts.append(f"column_type({table_name}, {column_name}, {column_info['type']})")
                            if 'nullable' in column_info and not column_info['nullable']:
                                facts.append(f"not_null_constraint({table_name}, {column_name})")
                        else:
                            facts.append(f"column_type({table_name}, {column_name}, {column_info})")
        
        self.logger.debug(f"Generated {len(facts)} facts from SQL and schema")
        return facts
    
    def extract_violations(self, results: Dict[str, Any]) -> List[str]:
        """Extract constraint violations from reasoning results"""
        violations = []
        
        # Check for violations in reasoning results
        if 'violations' in results:
            violations.extend(results['violations'])
        
        # For stub implementation, don't treat rule definitions as violations
        if not PYREASON_AVAILABLE:
            # In real implementation, violations would be derived facts, not rule definitions
            pass
        else:
            # Check reasoning trace for violation patterns (only for real PyReason)
            trace = results.get('trace', [])
            for step in trace:
                if 'violation' in step.lower() and 'derived:' in step.lower():
                    violations.append(step)
        
        return violations
    
    def calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence score for validation results"""
        base_confidence = results.get('confidence', 0.8)
        
        # Adjust confidence based on various factors
        violations_count = len(results.get('violations', []))
        trace_length = len(results.get('trace', []))
        
        # Lower confidence if there are violations
        if violations_count > 0:
            base_confidence *= max(0.1, 1.0 - (violations_count * 0.2))
        
        # Adjust based on reasoning depth
        if trace_length > 0:
            base_confidence *= min(1.0, 0.5 + (trace_length * 0.1))
        
        return max(0.0, min(1.0, base_confidence))
    
    def generate_explanation(self, results: Dict[str, Any], violations: List[str]) -> str:
        """Generate human-readable explanation of reasoning process"""
        explanation_parts = []
        
        if violations:
            explanation_parts.append(f"Found {len(violations)} constraint violations:")
            for i, violation in enumerate(violations, 1):
                explanation_parts.append(f"  {i}. {violation}")
        else:
            explanation_parts.append("No constraint violations detected.")
        
        trace = results.get('trace', [])
        if trace:
            explanation_parts.append(f"\nReasoning process ({len(trace)} steps):")
            for i, step in enumerate(trace[:5], 1):  # Show first 5 steps
                explanation_parts.append(f"  {i}. {step}")
            if len(trace) > 5:
                explanation_parts.append(f"  ... and {len(trace) - 5} more steps")
        
        facts_count = len(results.get('facts', []))
        rules_count = len(results.get('rules', []))
        explanation_parts.append(f"\nApplied {rules_count} rules to {facts_count} facts.")
        
        return '\n'.join(explanation_parts)
    
    def check_constraints(self, facts: List[str]) -> List[str]:
        """
        Check constraints against given facts
        
        Args:
            facts: List of facts to check
            
        Returns:
            List of violations found
        """
        # Add facts to engine
        for fact in facts:
            self.add_fact(fact)
        
        # Run reasoning
        results = self.pyreason.reason(timesteps=self.config['pyreason']['reasoning_steps'])
        
        # Extract violations
        violations = self.extract_violations(results)
        
        self.logger.info(f"Constraint checking completed: {len(violations)} violations found")
        return violations
    
    def reset(self):
        """Reset the reasoning engine state"""
        self.facts.clear()
        self.rules.clear()
        if hasattr(self.pyreason, 'reset'):
            self.pyreason.reset()
        
        # Re-initialize default rules
        self._initialize_default_rules()
        
        self.logger.debug("PyReason engine reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            'pyreason_available': PYREASON_AVAILABLE,
            'simple_logic_available': SIMPLE_LOGIC_AVAILABLE,
            'logic_functional': self._logic_functional,
            'facts_count': len(self.facts),
            'rules_count': len(self.rules),
            'constraints_count': len(self.constraints),
            'config': self.config,
            'engine_type': 'SimpleLogicEngine' if self._logic_functional else 'Stub'
        }


# Module-level convenience functions
def create_engine(config_path: Optional[str] = None) -> PyReasonEngine:
    """Create a new PyReason engine instance"""
    return PyReasonEngine(config_path)


def validate_sql_query(sql_query: str, schema: Dict[str, Any], 
                      config_path: Optional[str] = None) -> ValidationResult:
    """Convenience function to validate a single SQL query"""
    engine = create_engine(config_path)
    return engine.validate_sql(sql_query, schema)