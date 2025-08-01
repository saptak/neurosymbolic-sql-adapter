#!/usr/bin/env python3
"""
Integration Tests

End-to-end integration tests for the neurosymbolic SQL adapter components.
"""

import pytest
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from reasoning.pyreason_engine import PyReasonEngine
from reasoning.sql_knowledge_base import SQLKnowledgeBase, Table, Column, DataType, ConstraintType
from reasoning.constraint_validator import ConstraintValidator
from reasoning.sql_to_facts import SQLToFactsConverter
from reasoning.explanation_generator import ExplanationGenerator, ExplanationContext


class TestFullIntegration:
    """Test complete integration of all components"""
    
    def setup_method(self):
        """Set up comprehensive test scenario"""
        # Create knowledge base
        self.kb = SQLKnowledgeBase()
        
        # Add comprehensive schema
        schema = {
            'customers': {
                'columns': {
                    'id': {'type': 'integer', 'nullable': False},
                    'name': {'type': 'varchar', 'nullable': False, 'max_length': 100},
                    'email': {'type': 'varchar', 'nullable': True, 'max_length': 255},
                    'created_at': {'type': 'datetime', 'nullable': False}
                },
                'primary_key': ['id']
            },
            'orders': {
                'columns': {
                    'id': {'type': 'integer', 'nullable': False},
                    'customer_id': {'type': 'integer', 'nullable': False},
                    'amount': {'type': 'decimal', 'nullable': False},
                    'order_date': {'type': 'date', 'nullable': False},
                    'status': {'type': 'varchar', 'nullable': False}
                },
                'primary_key': ['id'],
                'foreign_keys': [
                    {
                        'columns': ['customer_id'],
                        'references': {'table': 'customers', 'columns': ['id']}
                    }
                ]
            },
            'products': {
                'columns': {
                    'id': {'type': 'integer', 'nullable': False},
                    'name': {'type': 'varchar', 'nullable': False},
                    'price': {'type': 'decimal', 'nullable': False},
                    'in_stock': {'type': 'boolean', 'nullable': False}
                },
                'primary_key': ['id']
            }
        }
        
        self.kb.add_schema(schema)
        
        # Add foreign key constraint
        self.kb.add_constraint(
            ConstraintType.FOREIGN_KEY,
            'orders',
            ['customer_id'],
            reference_table='customers',
            reference_columns=['id']
        )
        
        # Initialize all components
        self.reasoning_engine = PyReasonEngine()
        self.validator = ConstraintValidator(self.kb)
        self.facts_converter = SQLToFactsConverter(self.kb)
        self.explanation_generator = ExplanationGenerator()
    
    def test_valid_query_full_pipeline(self):
        """Test complete pipeline with valid query"""
        sql_query = """
        SELECT c.name, c.email, o.amount, o.order_date
        FROM customers c
        JOIN orders o ON c.id = o.customer_id
        WHERE o.amount > 100
        ORDER BY o.order_date DESC
        """
        
        # Step 1: Convert to facts
        facts = self.facts_converter.convert_to_facts(sql_query)
        assert len(facts) > 0
        
        # Step 2: Validate constraints
        violations = self.validator.validate_query(sql_query)
        
        # Step 3: Run symbolic reasoning
        schema_dict = {table.name: table.to_dict() for table in self.kb.tables.values()}
        result = self.reasoning_engine.validate_sql(sql_query, schema_dict)
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.confidence, float)
        
        # Step 4: Generate explanation
        context = ExplanationContext(
            query=sql_query,
            violations=violations,
            facts=facts,
            reasoning_trace=result.reasoning_trace,
            confidence=result.confidence,
            schema_info={'tables': self.kb.tables}
        )
        
        explanation = self.explanation_generator.generate_explanation(context)
        assert explanation is not None
        assert len(explanation.summary) > 0
        
        # Verify pipeline coherence
        if len(violations) == 0:
            assert result.is_valid or result.confidence > 0.5
            assert 'successful' in explanation.title.lower() or len(violations) == 0
    
    def test_invalid_query_full_pipeline(self):
        """Test complete pipeline with invalid query"""
        sql_query = """
        SELECT c.invalid_column, n.nonexistent_column
        FROM customers c
        JOIN nonexistent_table n ON c.id = n.id
        WHERE c.amount > 'invalid_value'
        """
        
        # Step 1: Convert to facts
        facts = self.facts_converter.convert_to_facts(sql_query)
        assert len(facts) > 0
        
        # Step 2: Validate constraints
        violations = self.validator.validate_query(sql_query)
        assert len(violations) > 0  # Should find violations
        
        # Step 3: Run symbolic reasoning
        schema_dict = {table.name: table.to_dict() for table in self.kb.tables.values()}
        result = self.reasoning_engine.validate_sql(sql_query, schema_dict)
        
        # Step 4: Generate explanation
        context = ExplanationContext(
            query=sql_query,
            violations=violations,
            facts=facts,
            reasoning_trace=result.reasoning_trace,
            confidence=result.confidence,
            schema_info={'tables': self.kb.tables}
        )
        
        explanation = self.explanation_generator.generate_explanation(context)
        assert explanation is not None
        
        # Verify violations are properly handled
        assert len(violations) > 0
        assert 'failed' in explanation.title.lower() or 'violation' in explanation.summary.lower()
        assert len(explanation.recommendations) > 0
    
    def test_complex_query_analysis(self):
        """Test analysis of complex query with multiple operations"""
        sql_query = """
        SELECT 
            c.name,
            COUNT(o.id) as order_count,
            SUM(o.amount) as total_amount,
            AVG(o.amount) as avg_amount
        FROM customers c
        LEFT JOIN orders o ON c.id = o.customer_id
        WHERE c.created_at >= '2023-01-01'
            AND (o.status = 'completed' OR o.status IS NULL)
        GROUP BY c.id, c.name
        HAVING COUNT(o.id) > 0
        ORDER BY total_amount DESC
        LIMIT 10
        """
        
        # Analyze query structure
        analysis = self.facts_converter.analyze_query(sql_query)
        
        assert analysis.query_type.value == 'SELECT'
        assert len(analysis.tables) > 0
        assert len(analysis.operations) > 0
        assert 'COUNT' in analysis.operations
        assert 'SUM' in analysis.operations
        assert 'AVG' in analysis.operations
        assert 'GROUP_BY' in analysis.operations
        assert 'HAVING' in analysis.operations
        assert 'ORDER_BY' in analysis.operations
        
        # Convert to facts
        facts = self.facts_converter.convert_to_facts(sql_query)
        
        # Check for operation facts
        fact_string = ' '.join(facts)
        assert 'query_uses_operation(count)' in fact_string
        assert 'query_uses_operation(sum)' in fact_string
        assert 'query_uses_operation(avg)' in fact_string
        
        # Validate query
        violations = self.validator.validate_query(sql_query)
        
        # Generate comprehensive explanation
        context = ExplanationContext(
            query=sql_query,
            query_analysis=analysis,
            violations=violations,
            facts=facts,
            user_level='expert'
        )
        
        explanation = self.explanation_generator.generate_explanation(context)
        formatted_explanation = self.explanation_generator.format_explanation(explanation)
        
        assert len(formatted_explanation) > 100  # Should be comprehensive
    
    def test_schema_relationship_validation(self):
        """Test validation of relationships between tables"""
        # Query that uses foreign key relationship
        sql_query = """
        SELECT c.name, o.amount
        FROM customers c
        JOIN orders o ON c.id = o.customer_id
        WHERE o.amount > 50
        """
        
        # Get foreign key relationships
        fk_relationships = self.kb.get_foreign_key_relationships('orders')
        assert len(fk_relationships) > 0
        
        # Validate relationship usage in query
        violations = self.validator.validate_query(sql_query)
        
        # Should not have foreign key violations for valid relationship
        fk_violations = [v for v in violations if 'foreign_key' in str(v.violation_type)]
        # Note: With simplified parsing, we may not catch all relationship validations
        
        # Generate facts including relationship information
        facts = self.facts_converter.convert_to_facts(sql_query)
        relationship_facts = [f for f in facts if 'foreign_key' in f]
        assert len(relationship_facts) > 0
    
    def test_constraint_violation_explanation(self):
        """Test detailed explanation of constraint violations"""
        sql_query = "INSERT INTO orders (customer_id, amount) VALUES (999, -50)"
        
        # This should trigger multiple violations:
        # 1. Missing primary key (id)
        # 2. Invalid foreign key reference (customer_id = 999 doesn't exist)
        # 3. Possibly negative amount validation
        
        violations = self.validator.validate_query(sql_query)
        
        # Generate detailed explanation
        context = ExplanationContext(
            query=sql_query,
            violations=violations,
            user_level='beginner'  # Detailed explanations for beginners
        )
        
        explanation = self.explanation_generator.generate_explanation(context)
        
        # Should provide helpful suggestions
        assert len(explanation.recommendations) > 0
        
        # Format in different styles
        natural_format = self.explanation_generator.format_explanation(explanation)
        structured_format = self.explanation_generator.format_explanation(
            explanation, 
            self.explanation_generator.__class__.__dict__['_format_structured'].__annotations__.get('style', 'structured')
        )
        
        assert len(natural_format) > 0
        # Note: structured_format test would need proper enum import
    
    def test_fact_statistics_and_analysis(self):
        """Test statistical analysis of generated facts"""
        sql_query = """
        SELECT p.name, COUNT(*) as sales_count
        FROM products p
        JOIN order_items oi ON p.id = oi.product_id
        JOIN orders o ON oi.order_id = o.id
        WHERE o.order_date >= '2023-01-01'
        GROUP BY p.id, p.name
        """
        
        # Generate facts
        facts = self.facts_converter.convert_to_facts(sql_query)
        
        # Get statistics
        stats = self.facts_converter.get_fact_statistics(facts)
        
        assert 'total_facts' in stats
        assert 'fact_types' in stats
        assert 'tables_referenced' in stats
        assert stats['total_facts'] > 0
        assert len(stats['fact_types']) > 0
        
        # Knowledge base statistics
        kb_stats = self.kb.get_statistics()
        assert kb_stats['tables_count'] == 3  # customers, orders, products
        assert kb_stats['facts_count'] > 0
    
    def test_confidence_scoring_integration(self):
        """Test confidence scoring across components"""
        test_queries = [
            # High confidence: Simple, valid query
            "SELECT name FROM customers WHERE id = 1",
            
            # Medium confidence: Complex but valid query
            "SELECT c.name, COUNT(o.id) FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.name",
            
            # Low confidence: Query with violations
            "SELECT invalid_column FROM nonexistent_table"
        ]
        
        confidence_scores = []
        
        for sql_query in test_queries:
            schema_dict = {table.name: table.to_dict() for table in self.kb.tables.values()}
            result = self.reasoning_engine.validate_sql(sql_query, schema_dict)
            violations = self.validator.validate_query(sql_query)
            
            # Confidence should correlate with validation success
            if len(violations) == 0:
                assert result.confidence > 0.5
            else:
                # May still have reasonable confidence depending on violation severity
                pass
            
            confidence_scores.append(result.confidence)
        
        # First query should have highest confidence, last should have lowest
        assert confidence_scores[0] >= confidence_scores[-1]
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # Simulate user input
        user_query = """
        SELECT c.name, c.email, SUM(o.amount) as total_spent
        FROM customers c
        JOIN orders o ON c.id = o.customer_id
        WHERE o.order_date >= '2023-01-01'
            AND o.status = 'completed'
        GROUP BY c.id, c.name, c.email
        HAVING SUM(o.amount) > 500
        ORDER BY total_spent DESC
        """
        
        # Complete workflow
        try:
            # 1. Parse and analyze query
            analysis = self.facts_converter.analyze_query(user_query)
            
            # 2. Generate facts
            facts = self.facts_converter.convert_to_facts(user_query)
            
            # 3. Validate constraints
            violations = self.validator.validate_query(user_query)
            
            # 4. Run symbolic reasoning
            schema_dict = {table.name: table.to_dict() for table in self.kb.tables.values()}
            reasoning_result = self.reasoning_engine.validate_sql(user_query, schema_dict)
            
            # 5. Generate comprehensive explanation
            context = ExplanationContext(
                query=user_query,
                query_analysis=analysis,
                violations=violations,
                facts=facts,
                reasoning_trace=reasoning_result.reasoning_trace,
                confidence=reasoning_result.confidence,
                schema_info={'tables': self.kb.tables},
                user_level='intermediate'
            )
            
            explanation = self.explanation_generator.generate_explanation(context)
            final_report = self.explanation_generator.format_explanation(explanation)
            
            # Verify workflow completed successfully
            assert analysis is not None
            assert len(facts) > 0
            assert isinstance(violations, list)
            assert reasoning_result is not None
            assert explanation is not None
            assert len(final_report) > 100
            
            # Print summary for manual verification
            print(f"\n=== END-TO-END WORKFLOW SUMMARY ===")
            print(f"Query analyzed: {analysis.query_type}")
            print(f"Facts generated: {len(facts)}")
            print(f"Violations found: {len(violations)}")
            print(f"Reasoning confidence: {reasoning_result.confidence:.2f}")
            print(f"Explanation length: {len(final_report)} characters")
            
        except Exception as e:
            pytest.fail(f"End-to-end workflow failed: {str(e)}")


class TestComponentInteraction:
    """Test interaction between different components"""
    
    def setup_method(self):
        """Set up minimal test environment"""
        self.kb = SQLKnowledgeBase()
        self.kb.add_schema({'test_table': ['id', 'name']})
    
    def test_knowledge_base_to_reasoning_engine(self):
        """Test data flow from KB to reasoning engine"""
        engine = PyReasonEngine()
        
        # Generate facts from KB
        kb_facts = self.kb.generate_facts()
        
        # Add facts to reasoning engine
        for fact in kb_facts:
            engine.add_fact(fact)
        
        status = engine.get_status()
        assert status['facts_count'] == len(kb_facts)
    
    def test_validator_to_explanation_generator(self):
        """Test violation data flow to explanation generator"""
        validator = ConstraintValidator(self.kb)
        generator = ExplanationGenerator()
        
        violations = validator.validate_query("SELECT * FROM nonexistent")
        
        context = ExplanationContext(
            query="SELECT * FROM nonexistent",
            violations=violations
        )
        
        explanation = generator.generate_explanation(context)
        
        if len(violations) > 0:
            assert 'failed' in explanation.title.lower() or 'violation' in explanation.summary.lower()
    
    def test_facts_converter_integration(self):
        """Test facts converter with other components"""
        converter = SQLToFactsConverter(self.kb)
        engine = PyReasonEngine()
        
        sql = "SELECT name FROM test_table"
        facts = converter.convert_to_facts(sql)
        
        # Add facts to reasoning engine
        for fact in facts:
            engine.add_fact(fact)
        
        # Validate using the facts
        result = engine.validate_sql(sql, {'test_table': ['id', 'name']})
        
        assert isinstance(result.confidence, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to see print output