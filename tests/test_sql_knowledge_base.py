#!/usr/bin/env python3
"""
Tests for SQLKnowledgeBase

Comprehensive test suite for the SQL knowledge base.
"""

import pytest
import json
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from reasoning.sql_knowledge_base import (
    SQLKnowledgeBase, Table, Column, DataType, ConstraintType, Relationship,
    create_knowledge_base, create_simple_schema
)


class TestSQLKnowledgeBase:
    """Test cases for SQLKnowledgeBase"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.kb = SQLKnowledgeBase()
        
        # Create test table
        self.test_table = Table(name='customers')
        self.test_table.add_column(Column(name='id', data_type=DataType.INTEGER, nullable=False))
        self.test_table.add_column(Column(name='name', data_type=DataType.VARCHAR, nullable=False))
        self.test_table.add_column(Column(name='email', data_type=DataType.VARCHAR, nullable=True))
        self.test_table.primary_key = ['id']
    
    def test_kb_initialization(self):
        """Test knowledge base initialization"""
        assert self.kb is not None
        assert len(self.kb.tables) == 0
        assert len(self.kb.relationships) == 0
        assert len(self.kb.constraints) == 0
        assert self.kb.schema_graph is not None
        assert self.kb.relationship_graph is not None
    
    def test_add_table(self):
        """Test adding tables to knowledge base"""
        self.kb.add_table(self.test_table)
        
        assert 'customers' in self.kb.tables
        assert len(self.kb.tables) == 1
        
        # Check if table was added to graph
        assert 'customers' in self.kb.schema_graph.nodes
        
        # Check if columns were added
        for column_name in ['id', 'name', 'email']:
            node_name = f"customers.{column_name}"
            assert node_name in self.kb.schema_graph.nodes
    
    def test_add_schema_simple(self):
        """Test adding simple schema format"""
        schema = {
            'customers': ['id', 'name', 'email'],
            'orders': ['id', 'customer_id', 'amount']
        }
        
        self.kb.add_schema(schema)
        
        assert len(self.kb.tables) == 2
        assert 'customers' in self.kb.tables
        assert 'orders' in self.kb.tables
        
        # Check columns
        customers_table = self.kb.get_table('customers')
        assert 'id' in customers_table.columns
        assert 'name' in customers_table.columns
        assert 'email' in customers_table.columns
    
    def test_add_schema_detailed(self):
        """Test adding detailed schema format"""
        schema = {
            'customers': {
                'columns': {
                    'id': {'type': 'integer', 'nullable': False},
                    'name': {'type': 'varchar', 'nullable': False, 'max_length': 100},
                    'email': {'type': 'varchar', 'nullable': True}
                },
                'primary_key': ['id']
            }
        }
        
        self.kb.add_schema(schema)
        
        customers_table = self.kb.get_table('customers')
        assert customers_table is not None
        assert customers_table.primary_key == ['id']
        
        id_column = customers_table.get_column('id')
        assert id_column.data_type == DataType.INTEGER
        assert id_column.nullable is False
        
        name_column = customers_table.get_column('name')
        assert name_column.max_length == 100
    
    def test_add_relationship(self):
        """Test adding relationships"""
        # Add tables first
        customers_table = Table(name='customers')
        customers_table.add_column(Column(name='id', data_type=DataType.INTEGER))
        
        orders_table = Table(name='orders')
        orders_table.add_column(Column(name='customer_id', data_type=DataType.INTEGER))
        
        self.kb.add_table(customers_table)
        self.kb.add_table(orders_table)
        
        # Add relationship
        relationship = Relationship(
            from_table='orders',
            from_column='customer_id',
            to_table='customers',
            to_column='id',
            relationship_type='foreign_key'
        )
        
        self.kb.add_relationship(relationship)
        
        assert len(self.kb.relationships) == 1
        assert self.kb.relationship_graph.has_edge('orders', 'customers')
    
    def test_add_constraint(self):
        """Test adding constraints"""
        self.kb.add_table(self.test_table)
        
        constraint_id = self.kb.add_constraint(
            ConstraintType.PRIMARY_KEY,
            'customers',
            ['id'],
            condition='unique(customers, id)'
        )
        
        assert constraint_id in self.kb.constraints
        constraint = self.kb.constraints[constraint_id]
        assert constraint['type'] == 'primary_key'
        assert constraint['table'] == 'customers'
        assert constraint['columns'] == ['id']
    
    def test_add_foreign_key_constraint(self):
        """Test adding foreign key constraints"""
        # Add both tables
        customers_table = Table(name='customers')
        customers_table.add_column(Column(name='id', data_type=DataType.INTEGER))
        
        orders_table = Table(name='orders')
        orders_table.add_column(Column(name='customer_id', data_type=DataType.INTEGER))
        
        self.kb.add_table(customers_table)
        self.kb.add_table(orders_table)
        
        # Add foreign key constraint
        constraint_id = self.kb.add_constraint(
            ConstraintType.FOREIGN_KEY,
            'orders',
            ['customer_id'],
            reference_table='customers',
            reference_columns=['id']
        )
        
        assert constraint_id in self.kb.constraints
        # Should also create a relationship
        assert len(self.kb.relationships) == 1
    
    def test_get_table(self):
        """Test retrieving tables"""
        self.kb.add_table(self.test_table)
        
        retrieved_table = self.kb.get_table('customers')
        assert retrieved_table is not None
        assert retrieved_table.name == 'customers'
        
        non_existent = self.kb.get_table('nonexistent')
        assert non_existent is None
    
    def test_get_related_tables(self):
        """Test getting related tables"""
        # Add tables and relationship
        customers_table = Table(name='customers')
        orders_table = Table(name='orders')
        
        self.kb.add_table(customers_table)
        self.kb.add_table(orders_table)
        
        relationship = Relationship(
            from_table='orders',
            from_column='customer_id',
            to_table='customers',
            to_column='id'
        )
        self.kb.add_relationship(relationship)
        
        related_to_customers = self.kb.get_related_tables('customers')
        assert 'orders' in related_to_customers
        
        related_to_orders = self.kb.get_related_tables('orders')
        assert 'customers' in related_to_orders
    
    def test_validate_references(self):
        """Test reference validation"""
        self.kb.add_table(self.test_table)
        
        # Valid references
        assert self.kb.validate_table_reference('customers') is True
        assert self.kb.validate_column_reference('customers', 'id') is True
        assert self.kb.validate_column_reference('customers', 'name') is True
        
        # Invalid references
        assert self.kb.validate_table_reference('nonexistent') is False
        assert self.kb.validate_column_reference('customers', 'invalid') is False
        assert self.kb.validate_column_reference('nonexistent', 'id') is False
    
    def test_get_column_type(self):
        """Test getting column data types"""
        self.kb.add_table(self.test_table)
        
        id_type = self.kb.get_column_type('customers', 'id')
        assert id_type == DataType.INTEGER
        
        name_type = self.kb.get_column_type('customers', 'name')
        assert name_type == DataType.VARCHAR
        
        # Non-existent column
        invalid_type = self.kb.get_column_type('customers', 'invalid')
        assert invalid_type is None
    
    def test_get_primary_key_columns(self):
        """Test getting primary key columns"""
        self.kb.add_table(self.test_table)
        
        pk_columns = self.kb.get_primary_key_columns('customers')
        assert pk_columns == ['id']
        
        # Table without primary key
        no_pk_table = Table(name='temp')
        self.kb.add_table(no_pk_table)
        pk_columns = self.kb.get_primary_key_columns('temp')
        assert pk_columns == []
    
    def test_generate_facts(self):
        """Test logical facts generation"""
        self.kb.add_table(self.test_table)
        
        facts = self.kb.generate_facts()
        
        assert isinstance(facts, list)
        assert len(facts) > 0
        
        # Check for expected facts
        fact_strings = ' '.join(facts)
        assert 'table_exists(customers)' in fact_strings
        assert 'column_exists(customers, id)' in fact_strings
        assert 'column_type(customers, id, integer)' in fact_strings
        assert 'not_null_constraint(customers, id)' in fact_strings
        assert 'not_null_constraint(customers, name)' in fact_strings
        assert 'primary_key(customers, id)' in fact_strings
    
    def test_export_schema(self):
        """Test schema export"""
        self.kb.add_table(self.test_table)
        
        # Export as JSON
        json_export = self.kb.export_schema('json')
        assert isinstance(json_export, str)
        
        # Parse to verify valid JSON
        schema_dict = json.loads(json_export)
        assert 'tables' in schema_dict
        assert 'customers' in schema_dict['tables']
        
        # Test invalid format
        with pytest.raises(ValueError):
            self.kb.export_schema('invalid_format')
    
    def test_import_schema(self):
        """Test schema import"""
        # Create schema data
        schema_data = {
            'tables': {
                'test_table': {
                    'name': 'test_table',
                    'columns': {
                        'id': {
                            'name': 'id',
                            'data_type': 'integer',
                            'nullable': False
                        }
                    },
                    'primary_key': ['id']
                }
            }
        }
        
        json_data = json.dumps(schema_data)
        self.kb.import_schema(json_data, 'json')
        
        assert 'test_table' in self.kb.tables
        test_table = self.kb.get_table('test_table')
        assert test_table.primary_key == ['id']
    
    def test_get_statistics(self):
        """Test getting knowledge base statistics"""
        self.kb.add_table(self.test_table)
        
        stats = self.kb.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'tables_count' in stats
        assert 'columns_count' in stats
        assert 'relationships_count' in stats
        assert 'constraints_count' in stats
        assert 'facts_count' in stats
        
        assert stats['tables_count'] == 1
        assert stats['columns_count'] == 3  # id, name, email
    
    def test_reset(self):
        """Test knowledge base reset"""
        self.kb.add_table(self.test_table)
        
        # Verify data exists
        assert len(self.kb.tables) > 0
        
        # Reset
        self.kb.reset()
        
        # Verify data is cleared
        assert len(self.kb.tables) == 0
        assert len(self.kb.relationships) == 0
        assert len(self.kb.constraints) == 0
        assert self.kb.schema_graph.number_of_nodes() == 0


class TestColumn:
    """Test cases for Column dataclass"""
    
    def test_column_creation(self):
        """Test Column creation"""
        column = Column(
            name='test_col',
            data_type=DataType.VARCHAR,
            nullable=False,
            max_length=100
        )
        
        assert column.name == 'test_col'
        assert column.data_type == DataType.VARCHAR
        assert column.nullable is False
        assert column.max_length == 100
    
    def test_column_to_dict(self):
        """Test Column to dictionary conversion"""
        column = Column(
            name='test_col',
            data_type=DataType.INTEGER,
            nullable=True,
            description='Test column'
        )
        
        column_dict = column.to_dict()
        
        assert isinstance(column_dict, dict)
        assert column_dict['name'] == 'test_col'
        assert column_dict['data_type'] == 'integer'
        assert column_dict['nullable'] is True
        assert column_dict['description'] == 'Test column'


class TestTable:
    """Test cases for Table dataclass"""
    
    def test_table_creation(self):
        """Test Table creation"""
        table = Table(name='test_table')
        
        assert table.name == 'test_table'
        assert len(table.columns) == 0
        assert table.primary_key is None
    
    def test_add_column(self):
        """Test adding columns to table"""
        table = Table(name='test_table')
        column = Column(name='id', data_type=DataType.INTEGER)
        
        table.add_column(column)
        
        assert 'id' in table.columns
        assert table.get_column('id') == column
    
    def test_table_to_dict(self):
        """Test Table to dictionary conversion"""
        table = Table(name='test_table')
        column = Column(name='id', data_type=DataType.INTEGER)
        table.add_column(column)
        table.primary_key = ['id']
        
        table_dict = table.to_dict()
        
        assert isinstance(table_dict, dict)
        assert table_dict['name'] == 'test_table'
        assert 'columns' in table_dict
        assert 'id' in table_dict['columns']
        assert table_dict['primary_key'] == ['id']


class TestDataType:
    """Test cases for DataType enum"""
    
    def test_data_type_values(self):
        """Test DataType enum values"""
        assert DataType.INTEGER.value == 'integer'
        assert DataType.VARCHAR.value == 'varchar'
        assert DataType.TEXT.value == 'text'
        assert DataType.BOOLEAN.value == 'boolean'
        assert DataType.DATE.value == 'date'


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_create_knowledge_base(self):
        """Test create_knowledge_base function"""
        kb = create_knowledge_base()
        assert isinstance(kb, SQLKnowledgeBase)
    
    def test_create_simple_schema(self):
        """Test create_simple_schema function"""
        tables_dict = {
            'customers': ['id', 'name', 'email'],
            'orders': ['id', 'customer_id', 'amount']
        }
        
        kb = create_simple_schema(tables_dict)
        assert isinstance(kb, SQLKnowledgeBase)
        assert len(kb.tables) == 2
        assert 'customers' in kb.tables
        assert 'orders' in kb.tables


if __name__ == "__main__":
    pytest.main([__file__, "-v"])