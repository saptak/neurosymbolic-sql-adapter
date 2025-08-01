#!/usr/bin/env python3
"""
Tests for PyReasonEngine

Comprehensive test suite for the symbolic reasoning engine.
"""

import pytest
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from reasoning.pyreason_engine import (
    PyReasonEngine, ValidationResult, Constraint, create_engine, validate_sql_query
)


class TestPyReasonEngine:
    """Test cases for PyReasonEngine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = PyReasonEngine()
        self.test_schema = {
            'customers': {
                'id': {'type': 'integer', 'nullable': False},
                'name': {'type': 'varchar', 'nullable': False},
                'email': {'type': 'varchar', 'nullable': True}
            },
            'orders': {
                'id': {'type': 'integer', 'nullable': False},
                'customer_id': {'type': 'integer', 'nullable': False},
                'amount': {'type': 'decimal', 'nullable': False}
            }
        }
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        assert self.engine is not None
        status = self.engine.get_status()
        assert 'pyreason_available' in status
        assert 'facts_count' in status
        assert 'rules_count' in status
        assert status['rules_count'] > 0  # Should have default rules
    
    def test_add_fact(self):
        """Test adding facts to the engine"""
        initial_count = len(self.engine.facts)
        self.engine.add_fact("table_exists(customers)")
        assert len(self.engine.facts) == initial_count + 1
        assert "table_exists(customers)" in self.engine.facts
    
    def test_add_rule(self):
        """Test adding rules to the engine"""
        initial_count = len(self.engine.rules)
        self.engine.add_rule("test_rule(X) :- condition(X)")
        assert len(self.engine.rules) == initial_count + 1
        assert "test_rule(X) :- condition(X)" in self.engine.rules
    
    def test_add_constraint(self):
        """Test adding constraints"""
        constraint = Constraint(
            constraint_type="primary_key",
            scope=["customers", "id"],
            condition="unique(customers, id)"
        )
        
        initial_count = len(self.engine.constraints)
        self.engine.add_constraint(constraint)
        assert len(self.engine.constraints) == initial_count + 1
    
    def test_validate_sql_simple(self):
        """Test basic SQL validation"""
        sql = "SELECT name FROM customers"
        result = self.engine.validate_sql(sql, self.test_schema)
        
        assert isinstance(result, ValidationResult)
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.confidence, float)
        assert isinstance(result.violations, list)
        assert isinstance(result.reasoning_trace, list)
    
    def test_validate_sql_with_joins(self):
        """Test SQL validation with joins"""
        sql = """SELECT c.name, o.amount 
                 FROM customers c 
                 JOIN orders o ON c.id = o.customer_id"""
        
        result = self.engine.validate_sql(sql, self.test_schema)
        assert isinstance(result, ValidationResult)
        assert result.facts_applied is not None
        assert len(result.facts_applied) > 0
    
    def test_sql_to_facts_conversion(self):
        """Test SQL to facts conversion"""
        sql = "SELECT id, name FROM customers WHERE id = 1"
        facts = self.engine.sql_to_facts(sql, self.test_schema)
        
        assert isinstance(facts, list)
        assert len(facts) > 0
        # Should contain table and column references
        assert any("customers" in fact for fact in facts)
        assert any("id" in fact for fact in facts)
        assert any("name" in fact for fact in facts)
    
    def test_constraint_checking(self):
        """Test constraint checking functionality"""
        test_facts = [
            "primary_key(customers, id)",
            "foreign_key(orders, customer_id, customers, id)",
            "not_null_constraint(customers, name)"
        ]
        
        violations = self.engine.check_constraints(test_facts)
        assert isinstance(violations, list)
        # With stub implementation, should return empty list
        assert len(violations) == 0
    
    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        results = {
            'confidence': 0.9,
            'violations': [],
            'trace': ['step1', 'step2']
        }
        
        confidence = self.engine.calculate_confidence(results)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_explanation_generation(self):
        """Test explanation generation"""
        results = {
            'violations': [],
            'trace': ['Applied rule 1', 'Applied rule 2'],
            'facts': ['fact1', 'fact2'],
            'rules': ['rule1', 'rule2']
        }
        
        explanation = self.engine.generate_explanation(results, [])
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert 'No constraint violations detected' in explanation
    
    def test_engine_reset(self):
        """Test engine reset functionality"""
        # Add some facts and rules
        self.engine.add_fact("test_fact")
        self.engine.add_rule("test_rule")
        
        # Reset engine
        self.engine.reset()
        
        # Facts should be cleared, but default rules should remain
        assert len(self.engine.facts) == 0
        assert len(self.engine.rules) > 0  # Default rules
    
    def test_configuration_loading(self):
        """Test configuration loading"""
        config = self.engine.config
        assert isinstance(config, dict)
        assert 'pyreason' in config
        assert 'validation' in config
        assert 'reasoning_steps' in config['pyreason']
    
    def test_convenience_functions(self):
        """Test module-level convenience functions"""
        # Test create_engine
        engine = create_engine()
        assert isinstance(engine, PyReasonEngine)
        
        # Test validate_sql_query
        result = validate_sql_query("SELECT * FROM customers", self.test_schema)
        assert isinstance(result, ValidationResult)


class TestValidationResult:
    """Test cases for ValidationResult dataclass"""
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation"""
        result = ValidationResult(
            is_valid=True,
            violations=[],
            reasoning_trace=['step1', 'step2'],
            confidence=0.95,
            explanation="Test explanation"
        )
        
        assert result.is_valid is True
        assert result.violations == []
        assert result.reasoning_trace == ['step1', 'step2']
        assert result.confidence == 0.95
        assert result.explanation == "Test explanation"
    
    def test_validation_result_defaults(self):
        """Test ValidationResult with default values"""
        result = ValidationResult(
            is_valid=False,
            violations=['violation1'],
            reasoning_trace=[],
            confidence=0.5
        )
        
        assert result.explanation is None
        assert result.facts_applied is None
        assert result.rules_applied is None


class TestConstraint:
    """Test cases for Constraint dataclass"""
    
    def test_constraint_creation(self):
        """Test Constraint creation"""
        constraint = Constraint(
            constraint_type="primary_key",
            scope=["table", "column"],
            condition="unique(table, column)"
        )
        
        assert constraint.constraint_type == "primary_key"
        assert constraint.scope == ["table", "column"]
        assert constraint.condition == "unique(table, column)"
        assert constraint.severity == "error"  # Default
    
    def test_constraint_to_pyreason_rule(self):
        """Test constraint to PyReason rule conversion"""
        constraint = Constraint(
            constraint_type="primary_key",
            scope=["customers", "id"],
            condition="unique(customers, id)"
        )
        
        rule = constraint.to_pyreason_rule()
        assert isinstance(rule, str)
        assert "primary_key" in rule
        assert "customers" in rule
        assert "id" in rule
    
    def test_constraint_types(self):
        """Test different constraint types"""
        constraint_types = ["primary_key", "foreign_key", "not_null", "unique", "check"]
        
        for ctype in constraint_types:
            constraint = Constraint(
                constraint_type=ctype,
                scope=["table", "column"],
                condition="test_condition"
            )
            
            rule = constraint.to_pyreason_rule()
            assert ctype in rule


if __name__ == "__main__":
    pytest.main([__file__, "-v"])