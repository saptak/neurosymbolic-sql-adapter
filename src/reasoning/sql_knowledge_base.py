#!/usr/bin/env python3
"""
SQL Knowledge Base

Manages database schema relationships, constraints, and semantic mappings
for symbolic reasoning about SQL queries.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
import json


class ConstraintType(Enum):
    """Types of database constraints"""
    PRIMARY_KEY = "primary_key"
    FOREIGN_KEY = "foreign_key"
    UNIQUE = "unique"
    NOT_NULL = "not_null"
    CHECK = "check"
    DEFAULT = "default"
    INDEX = "index"


class DataType(Enum):
    """SQL data types"""
    INTEGER = "integer"
    VARCHAR = "varchar"
    TEXT = "text"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    TIMESTAMP = "timestamp"
    DECIMAL = "decimal"
    FLOAT = "float"
    JSON = "json"
    BLOB = "blob"


@dataclass
class Column:
    """Represents a database column"""
    name: str
    data_type: DataType
    nullable: bool = True
    default_value: Optional[str] = None
    max_length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    constraints: List[str] = field(default_factory=list)
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert column to dictionary representation"""
        return {
            'name': self.name,
            'data_type': self.data_type.value,
            'nullable': self.nullable,
            'default_value': self.default_value,
            'max_length': self.max_length,
            'precision': self.precision,
            'scale': self.scale,
            'constraints': self.constraints,
            'description': self.description
        }


@dataclass
class Table:
    """Represents a database table"""
    name: str
    columns: Dict[str, Column] = field(default_factory=dict)
    primary_key: Optional[List[str]] = None
    foreign_keys: List[Dict[str, Any]] = field(default_factory=list)
    indexes: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    description: Optional[str] = None
    
    def add_column(self, column: Column):
        """Add a column to the table"""
        self.columns[column.name] = column
    
    def get_column(self, name: str) -> Optional[Column]:
        """Get a column by name"""
        return self.columns.get(name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert table to dictionary representation"""
        return {
            'name': self.name,
            'columns': {name: col.to_dict() for name, col in self.columns.items()},
            'primary_key': self.primary_key,
            'foreign_keys': self.foreign_keys,
            'indexes': self.indexes,
            'constraints': self.constraints,
            'description': self.description
        }


@dataclass
class Relationship:
    """Represents a relationship between tables"""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    relationship_type: str = "foreign_key"  # foreign_key, one_to_one, one_to_many, many_to_many
    cardinality: str = "many_to_one"
    description: Optional[str] = None


class SQLKnowledgeBase:
    """
    SQL Knowledge Base for schema relationships and constraints
    
    Manages database schema information, relationships, and constraints
    for symbolic reasoning about SQL queries.
    """
    
    def __init__(self):
        """Initialize the SQL knowledge base"""
        self.logger = logging.getLogger(__name__)
        
        # Core data structures
        self.tables: Dict[str, Table] = {}
        self.relationships: List[Relationship] = []
        self.constraints: Dict[str, Dict[str, Any]] = {}
        self.semantic_relationships: Dict[str, List[str]] = {}
        
        # Graph representation
        self.schema_graph = nx.DiGraph()
        self.relationship_graph = nx.Graph()
        
        # Type mappings for data type inference
        self.type_mappings = self._initialize_type_mappings()
        
        self.logger.info("SQL Knowledge Base initialized")
    
    def _initialize_type_mappings(self) -> Dict[str, DataType]:
        """Initialize type mappings for common SQL types"""
        return {
            'int': DataType.INTEGER,
            'integer': DataType.INTEGER,
            'bigint': DataType.INTEGER,
            'smallint': DataType.INTEGER,
            'varchar': DataType.VARCHAR,
            'char': DataType.VARCHAR,
            'text': DataType.TEXT,
            'string': DataType.VARCHAR,
            'bool': DataType.BOOLEAN,
            'boolean': DataType.BOOLEAN,
            'date': DataType.DATE,
            'datetime': DataType.DATETIME,
            'timestamp': DataType.TIMESTAMP,
            'decimal': DataType.DECIMAL,
            'numeric': DataType.DECIMAL,
            'float': DataType.FLOAT,
            'real': DataType.FLOAT,
            'double': DataType.FLOAT,
            'json': DataType.JSON,
            'jsonb': DataType.JSON,
            'blob': DataType.BLOB,
            'binary': DataType.BLOB
        }
    
    def add_table(self, table: Table) -> None:
        """Add a table to the knowledge base"""
        self.tables[table.name] = table
        
        # Add to schema graph
        self.schema_graph.add_node(table.name, 
                                  type='table', 
                                  description=table.description)
        
        # Add columns as nodes
        for column_name, column in table.columns.items():
            column_node = f"{table.name}.{column_name}"
            self.schema_graph.add_node(column_node, 
                                     type='column',
                                     data_type=column.data_type.value,
                                     nullable=column.nullable,
                                     description=column.description)
            
            # Connect table to column
            self.schema_graph.add_edge(table.name, column_node, 
                                     relationship='contains')
        
        self.logger.debug(f"Added table: {table.name} with {len(table.columns)} columns")
    
    def add_schema(self, schema_dict: Dict[str, Any]) -> None:
        """
        Add database schema from dictionary format
        
        Args:
            schema_dict: Schema in various formats (simple list, detailed dict, etc.)
        """
        for table_name, table_info in schema_dict.items():
            table = Table(name=table_name)
            
            if isinstance(table_info, list):
                # Simple format: table_name: [column1, column2, ...]
                for column_name in table_info:
                    column = Column(name=column_name, data_type=DataType.VARCHAR)
                    table.add_column(column)
            
            elif isinstance(table_info, dict):
                # Detailed format with column information
                columns_info = table_info.get('columns', {})
                
                for column_name, column_info in columns_info.items():
                    if isinstance(column_info, str):
                        # Simple type specification
                        data_type = self._parse_data_type(column_info)
                        column = Column(name=column_name, data_type=data_type)
                    
                    elif isinstance(column_info, dict):
                        # Detailed column specification
                        data_type = self._parse_data_type(column_info.get('type', 'varchar'))
                        column = Column(
                            name=column_name,
                            data_type=data_type,
                            nullable=column_info.get('nullable', True),
                            default_value=column_info.get('default'),
                            max_length=column_info.get('max_length'),
                            description=column_info.get('description')
                        )
                    else:
                        # Fallback
                        column = Column(name=column_name, data_type=DataType.VARCHAR)
                    
                    table.add_column(column)
                
                # Add table-level constraints
                if 'primary_key' in table_info:
                    table.primary_key = table_info['primary_key']
                
                if 'foreign_keys' in table_info:
                    table.foreign_keys = table_info['foreign_keys']
                
                if 'constraints' in table_info:
                    table.constraints = table_info['constraints']
            
            self.add_table(table)
        
        self.logger.info(f"Added schema with {len(schema_dict)} tables")
    
    def _parse_data_type(self, type_str: str) -> DataType:
        """Parse data type string to DataType enum"""
        type_lower = type_str.lower().strip()
        
        # Handle common variations
        for type_key, data_type in self.type_mappings.items():
            if type_key in type_lower:
                return data_type
        
        # Default fallback
        return DataType.VARCHAR
    
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship between tables"""
        self.relationships.append(relationship)
        
        # Add to relationship graph
        self.relationship_graph.add_edge(
            relationship.from_table, 
            relationship.to_table,
            relationship_type=relationship.relationship_type,
            cardinality=relationship.cardinality,
            from_column=relationship.from_column,
            to_column=relationship.to_column,
            description=relationship.description
        )
        
        # Add to schema graph as well
        from_column = f"{relationship.from_table}.{relationship.from_column}"
        to_column = f"{relationship.to_table}.{relationship.to_column}"
        
        if from_column in self.schema_graph and to_column in self.schema_graph:
            self.schema_graph.add_edge(
                from_column, to_column,
                relationship=relationship.relationship_type,
                cardinality=relationship.cardinality
            )
        
        self.logger.debug(f"Added relationship: {relationship.from_table}.{relationship.from_column} -> {relationship.to_table}.{relationship.to_column}")
    
    def add_constraint(self, constraint_type: ConstraintType, 
                      table_name: str, columns: List[str], 
                      condition: Optional[str] = None,
                      reference_table: Optional[str] = None,
                      reference_columns: Optional[List[str]] = None) -> str:
        """
        Add a constraint to the knowledge base
        
        Returns:
            constraint_id: Unique identifier for the constraint
        """
        constraint_id = f"{constraint_type.value}_{table_name}_{len(self.constraints)}"
        
        constraint = {
            'id': constraint_id,
            'type': constraint_type.value,
            'table': table_name,
            'columns': columns,
            'condition': condition,
            'reference_table': reference_table,
            'reference_columns': reference_columns or []
        }
        
        self.constraints[constraint_id] = constraint
        
        # Add to table if it exists
        if table_name in self.tables:
            self.tables[table_name].constraints.append(constraint)
        
        # Create relationship for foreign keys
        if constraint_type == ConstraintType.FOREIGN_KEY and reference_table:
            for i, column in enumerate(columns):
                ref_column = reference_columns[i] if reference_columns and i < len(reference_columns) else column
                relationship = Relationship(
                    from_table=table_name,
                    from_column=column,
                    to_table=reference_table,
                    to_column=ref_column,
                    relationship_type="foreign_key"
                )
                self.add_relationship(relationship)
        
        self.logger.debug(f"Added constraint: {constraint_id}")
        return constraint_id
    
    def get_table(self, table_name: str) -> Optional[Table]:
        """Get a table by name"""
        return self.tables.get(table_name)
    
    def get_related_tables(self, table_name: str) -> List[str]:
        """Get tables related to the given table"""
        related = set()
        
        for relationship in self.relationships:
            if relationship.from_table == table_name:
                related.add(relationship.to_table)
            elif relationship.to_table == table_name:
                related.add(relationship.from_table)
        
        return list(related)
    
    def get_foreign_key_relationships(self, table_name: str) -> List[Relationship]:
        """Get foreign key relationships for a table"""
        fk_relationships = []
        
        for relationship in self.relationships:
            if (relationship.from_table == table_name or relationship.to_table == table_name) and \
               relationship.relationship_type == "foreign_key":
                fk_relationships.append(relationship)
        
        return fk_relationships
    
    def validate_column_reference(self, table_name: str, column_name: str) -> bool:
        """Validate if a column reference is valid"""
        table = self.get_table(table_name)
        return table is not None and column_name in table.columns
    
    def validate_table_reference(self, table_name: str) -> bool:
        """Validate if a table reference is valid"""
        return table_name in self.tables
    
    def get_column_type(self, table_name: str, column_name: str) -> Optional[DataType]:
        """Get the data type of a column"""
        table = self.get_table(table_name)
        if table:
            column = table.get_column(column_name)
            if column:
                return column.data_type
        return None
    
    def get_primary_key_columns(self, table_name: str) -> List[str]:
        """Get primary key columns for a table"""
        table = self.get_table(table_name)
        return table.primary_key if table and table.primary_key else []
    
    def generate_facts(self) -> List[str]:
        """Generate logical facts for symbolic reasoning"""
        facts = []
        
        # Table existence facts
        for table_name in self.tables:
            facts.append(f"table_exists({table_name})")
        
        # Column facts
        for table_name, table in self.tables.items():
            for column_name, column in table.columns.items():
                facts.append(f"column_exists({table_name}, {column_name})")
                facts.append(f"column_type({table_name}, {column_name}, {column.data_type.value})")
                
                if not column.nullable:
                    facts.append(f"not_null_constraint({table_name}, {column_name})")
        
        # Primary key facts
        for table_name, table in self.tables.items():
            if table.primary_key:
                for column in table.primary_key:
                    facts.append(f"primary_key({table_name}, {column})")
        
        # Foreign key facts
        for relationship in self.relationships:
            if relationship.relationship_type == "foreign_key":
                facts.append(f"foreign_key({relationship.from_table}, {relationship.from_column}, {relationship.to_table}, {relationship.to_column})")
        
        # Constraint facts
        for constraint_id, constraint in self.constraints.items():
            constraint_type = constraint['type']
            table = constraint['table']
            columns = constraint['columns']
            
            for column in columns:
                facts.append(f"{constraint_type}_constraint({table}, {column})")
        
        return facts
    
    def export_schema(self, format_type: str = "json") -> str:
        """Export schema in specified format"""
        if format_type.lower() == "json":
            schema_dict = {
                'tables': {name: table.to_dict() for name, table in self.tables.items()},
                'relationships': [
                    {
                        'from_table': rel.from_table,
                        'from_column': rel.from_column,
                        'to_table': rel.to_table,
                        'to_column': rel.to_column,
                        'type': rel.relationship_type,
                        'cardinality': rel.cardinality,
                        'description': rel.description
                    } for rel in self.relationships
                ],
                'constraints': self.constraints
            }
            return json.dumps(schema_dict, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def import_schema(self, schema_data: str, format_type: str = "json") -> None:
        """Import schema from specified format"""
        if format_type.lower() == "json":
            schema_dict = json.loads(schema_data)
            
            # Import tables
            if 'tables' in schema_dict:
                for table_name, table_data in schema_dict['tables'].items():
                    table = Table(name=table_name)
                    
                    # Import columns
                    if 'columns' in table_data:
                        for column_name, column_data in table_data['columns'].items():
                            data_type = DataType(column_data.get('data_type', 'varchar'))
                            column = Column(
                                name=column_name,
                                data_type=data_type,
                                nullable=column_data.get('nullable', True),
                                default_value=column_data.get('default_value'),
                                max_length=column_data.get('max_length'),
                                description=column_data.get('description')
                            )
                            table.add_column(column)
                    
                    # Import table metadata
                    table.primary_key = table_data.get('primary_key')
                    table.foreign_keys = table_data.get('foreign_keys', [])
                    table.constraints = table_data.get('constraints', [])
                    table.description = table_data.get('description')
                    
                    self.add_table(table)
            
            # Import relationships
            if 'relationships' in schema_dict:
                for rel_data in schema_dict['relationships']:
                    relationship = Relationship(
                        from_table=rel_data['from_table'],
                        from_column=rel_data['from_column'],
                        to_table=rel_data['to_table'],
                        to_column=rel_data['to_column'],
                        relationship_type=rel_data.get('type', 'foreign_key'),
                        cardinality=rel_data.get('cardinality', 'many_to_one'),
                        description=rel_data.get('description')
                    )
                    self.add_relationship(relationship)
            
            # Import constraints
            if 'constraints' in schema_dict:
                self.constraints.update(schema_dict['constraints'])
        
        else:
            raise ValueError(f"Unsupported import format: {format_type}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        total_columns = sum(len(table.columns) for table in self.tables.values())
        total_constraints = len(self.constraints)
        total_relationships = len(self.relationships)
        
        return {
            'tables_count': len(self.tables),
            'columns_count': total_columns,
            'relationships_count': total_relationships,
            'constraints_count': total_constraints,
            'facts_count': len(self.generate_facts()),
            'graph_nodes': self.schema_graph.number_of_nodes(),
            'graph_edges': self.schema_graph.number_of_edges()
        }
    
    def reset(self) -> None:
        """Reset the knowledge base"""
        self.tables.clear()
        self.relationships.clear()
        self.constraints.clear()
        self.semantic_relationships.clear()
        self.schema_graph.clear()
        self.relationship_graph.clear()
        
        self.logger.info("SQL Knowledge Base reset")


# Convenience functions
def create_knowledge_base() -> SQLKnowledgeBase:
    """Create a new SQL knowledge base instance"""
    return SQLKnowledgeBase()


def create_simple_schema(tables_dict: Dict[str, List[str]]) -> SQLKnowledgeBase:
    """Create knowledge base from simple table-column mapping"""
    kb = SQLKnowledgeBase()
    kb.add_schema(tables_dict)
    return kb