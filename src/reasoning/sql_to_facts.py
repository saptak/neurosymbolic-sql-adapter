#!/usr/bin/env python3
"""
SQL to Facts Converter

Converts SQL queries and database schemas to logical facts for symbolic reasoning.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import sqlparse
from sqlparse.sql import Statement, Token, Identifier, IdentifierList, Function, Where, Comparison
from sqlparse.tokens import Keyword, Name, Operator, Literal

from .sql_knowledge_base import SQLKnowledgeBase, DataType


class QueryType(Enum):
    """Types of SQL queries"""
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CREATE = "CREATE"
    ALTER = "ALTER"
    DROP = "DROP"


@dataclass
class QueryAnalysis:
    """Analysis of a SQL query"""
    query_type: QueryType
    tables: List[str]
    columns: List[Tuple[str, str]]  # (table, column) pairs
    conditions: List[str]
    joins: List[Dict[str, str]]
    operations: List[str]
    literals: List[Tuple[str, str]]  # (value, type) pairs


class SQLToFactsConverter:
    """
    Converts SQL queries and database schemas to logical facts
    
    Provides comprehensive conversion of SQL constructs to symbolic facts
    that can be used for reasoning about query correctness and constraints.
    """
    
    def __init__(self, knowledge_base: Optional[SQLKnowledgeBase] = None):
        """
        Initialize SQL to facts converter
        
        Args:
            knowledge_base: Optional knowledge base for schema information
        """
        self.logger = logging.getLogger(__name__)
        self.knowledge_base = knowledge_base
        
        # SQL pattern matching
        self.patterns = {
            'table_reference': re.compile(r'\bFROM\s+(\w+)', re.IGNORECASE),
            'column_reference': re.compile(r'(\w+)\.(\w+)', re.IGNORECASE),
            'join_pattern': re.compile(r'\b(\w+)\s+JOIN\s+(\w+)\s+ON\s+([^;]+)', re.IGNORECASE),
            'where_condition': re.compile(r'\bWHERE\s+([^;]+)', re.IGNORECASE),
            'literal_string': re.compile(r"'([^']*)'"),
            'literal_number': re.compile(r'\b(\d+(?:\.\d+)?)\b'),
            'comparison_op': re.compile(r'([<>=!]+)')
        }
        
        self.logger.info("SQL to Facts converter initialized")
    
    def analyze_query(self, sql_query: str) -> QueryAnalysis:
        """
        Analyze SQL query structure
        
        Args:
            sql_query: SQL query string
            
        Returns:
            QueryAnalysis with extracted components
        """
        sql_upper = sql_query.strip().upper()
        
        # Determine query type
        query_type = QueryType.SELECT  # default
        for qtype in QueryType:
            if sql_upper.startswith(qtype.value):
                query_type = qtype
                break
        
        try:
            # Parse with sqlparse
            parsed = sqlparse.parse(sql_query)[0]
            
            # Extract components
            tables = self._extract_tables(parsed, sql_query)
            columns = self._extract_column_references(parsed, sql_query)
            conditions = self._extract_conditions(parsed, sql_query)
            joins = self._extract_joins(parsed, sql_query)
            operations = self._extract_operations(parsed, sql_query)
            literals = self._extract_literals(parsed, sql_query)
            
            return QueryAnalysis(
                query_type=query_type,
                tables=tables,
                columns=columns,
                conditions=conditions,
                joins=joins,
                operations=operations,
                literals=literals
            )
        
        except Exception as e:
            self.logger.error(f"Error analyzing query: {e}")
            return QueryAnalysis(
                query_type=query_type,
                tables=[],
                columns=[],
                conditions=[],
                joins=[],
                operations=[],
                literals=[]
            )
    
    def _extract_tables(self, parsed: Statement, sql_query: str) -> List[str]:
        """Extract table names from parsed SQL"""
        tables = []
        
        # Look for FROM and JOIN clauses
        from_matches = self.patterns['table_reference'].findall(sql_query)
        tables.extend(from_matches)
        
        # Extract from JOIN clauses
        join_matches = self.patterns['join_pattern'].findall(sql_query)
        for match in join_matches:
            tables.extend([match[0], match[1]])
        
        # Parse using sqlparse tokens
        from_seen = False
        for token in parsed.flatten():
            if token.ttype is Keyword:
                if token.value.upper() == 'FROM':
                    from_seen = True
                    continue
                elif token.value.upper() in ['WHERE', 'GROUP', 'HAVING', 'ORDER', 'LIMIT']:
                    from_seen = False
                elif token.value.upper() == 'JOIN':
                    from_seen = True
                    continue
            
            if from_seen and token.ttype is Name and token.value not in ['ON', 'AS']:
                if token.value not in tables:
                    tables.append(token.value)
        
        return list(set(tables))  # Remove duplicates
    
    def _extract_column_references(self, parsed: Statement, sql_query: str) -> List[Tuple[str, str]]:
        """Extract column references (table.column)"""
        columns = []
        
        # Pattern matching for table.column
        column_matches = self.patterns['column_reference'].findall(sql_query)
        columns.extend(column_matches)
        
        # Parse tokens for column references
        for token in parsed.flatten():
            if token.ttype is Name and '.' in str(token):
                parts = str(token).split('.')
                if len(parts) == 2:
                    columns.append((parts[0], parts[1]))
        
        return list(set(columns))  # Remove duplicates
    
    def _extract_conditions(self, parsed: Statement, sql_query: str) -> List[str]:
        """Extract WHERE conditions and other predicates"""
        conditions = []
        
        # Extract WHERE clause
        where_match = self.patterns['where_condition'].search(sql_query)
        if where_match:
            where_clause = where_match.group(1).strip()
            conditions.append(where_clause)
        
        # Extract JOIN conditions
        join_matches = self.patterns['join_pattern'].findall(sql_query)
        for match in join_matches:
            join_condition = match[2].strip()
            conditions.append(join_condition)
        
        return conditions
    
    def _extract_joins(self, parsed: Statement, sql_query: str) -> List[Dict[str, str]]:
        """Extract JOIN information"""
        joins = []
        
        join_matches = self.patterns['join_pattern'].findall(sql_query)
        for match in join_matches:
            join_info = {
                'left_table': match[0],
                'right_table': match[1],
                'condition': match[2],
                'type': 'INNER'  # Default, could be enhanced to detect LEFT, RIGHT, etc.
            }
            joins.append(join_info)
        
        return joins
    
    def _extract_operations(self, parsed: Statement, sql_query: str) -> List[str]:
        """Extract SQL operations and functions"""
        operations = []
        
        sql_upper = sql_query.upper()
        
        # Common SQL operations
        if 'COUNT(' in sql_upper:
            operations.append('COUNT')
        if 'SUM(' in sql_upper:
            operations.append('SUM')
        if 'AVG(' in sql_upper:
            operations.append('AVG')
        if 'MAX(' in sql_upper:
            operations.append('MAX')
        if 'MIN(' in sql_upper:
            operations.append('MIN')
        if 'GROUP BY' in sql_upper:
            operations.append('GROUP_BY')
        if 'ORDER BY' in sql_upper:
            operations.append('ORDER_BY')
        if 'HAVING' in sql_upper:
            operations.append('HAVING')
        
        return operations
    
    def _extract_literals(self, parsed: Statement, sql_query: str) -> List[Tuple[str, str]]:
        """Extract literal values and their types"""
        literals = []
        
        # String literals
        string_matches = self.patterns['literal_string'].findall(sql_query)
        for match in string_matches:
            literals.append((match, 'string'))
        
        # Numeric literals
        number_matches = self.patterns['literal_number'].findall(sql_query)
        for match in number_matches:
            if '.' in match:
                literals.append((match, 'float'))
            else:
                literals.append((match, 'integer'))
        
        return literals
    
    def convert_query_to_facts(self, sql_query: str) -> List[str]:
        """
        Convert SQL query to logical facts
        
        Args:
            sql_query: SQL query string
            
        Returns:
            List of logical facts representing the query
        """
        facts = []
        
        # Analyze query
        analysis = self.analyze_query(sql_query)
        
        # Query type fact
        facts.append(f"query_type({analysis.query_type.value.lower()})")
        
        # Table reference facts
        for table in analysis.tables:
            facts.append(f"query_references_table({table})")
        
        # Column reference facts
        for table, column in analysis.columns:
            facts.append(f"query_references_column({table}, {column})")
        
        # Join facts
        for join in analysis.joins:
            facts.append(f"query_has_join({join['left_table']}, {join['right_table']}, {join['type'].lower()})")
            facts.append(f"join_condition({join['left_table']}, {join['right_table']}, '{join['condition']}')")
        
        # Condition facts
        for i, condition in enumerate(analysis.conditions):
            facts.append(f"query_has_condition(condition_{i}, '{condition}')")
        
        # Operation facts
        for operation in analysis.operations:
            facts.append(f"query_uses_operation({operation.lower()})")
        
        # Literal facts
        for value, value_type in analysis.literals:
            facts.append(f"query_has_literal({value}, {value_type})")
        
        self.logger.debug(f"Generated {len(facts)} facts from query analysis")
        return facts
    
    def convert_schema_to_facts(self, schema: Dict[str, Any]) -> List[str]:
        """
        Convert database schema to logical facts
        
        Args:
            schema: Database schema dictionary
            
        Returns:
            List of logical facts representing the schema
        """
        facts = []
        
        if self.knowledge_base:
            # Use knowledge base to generate comprehensive facts
            return self.knowledge_base.generate_facts()
        
        # Fallback: simple schema conversion
        for table_name, table_info in schema.items():
            facts.append(f"table_exists({table_name})")
            
            if isinstance(table_info, list):
                # Simple column list
                for column in table_info:
                    facts.append(f"column_exists({table_name}, {column})")
                    facts.append(f"column_type({table_name}, {column}, varchar)")  # Default type
            
            elif isinstance(table_info, dict):
                # Detailed table information
                columns = table_info.get('columns', {})
                
                for column_name, column_info in columns.items():
                    facts.append(f"column_exists({table_name}, {column_name})")
                    
                    if isinstance(column_info, str):
                        facts.append(f"column_type({table_name}, {column_name}, {column_info})")
                    elif isinstance(column_info, dict):
                        col_type = column_info.get('type', 'varchar')
                        facts.append(f"column_type({table_name}, {column_name}, {col_type})")
                        
                        if not column_info.get('nullable', True):
                            facts.append(f"not_null_constraint({table_name}, {column_name})")
                
                # Primary key facts
                if 'primary_key' in table_info:
                    pk_columns = table_info['primary_key']
                    if isinstance(pk_columns, list):
                        for pk_col in pk_columns:
                            facts.append(f"primary_key({table_name}, {pk_col})")
                    else:
                        facts.append(f"primary_key({table_name}, {pk_columns})")
                
                # Foreign key facts
                if 'foreign_keys' in table_info:
                    for fk in table_info['foreign_keys']:
                        if 'columns' in fk and 'references' in fk:
                            fk_columns = fk['columns']
                            ref_table = fk['references']['table']
                            ref_columns = fk['references']['columns']
                            
                            for i, fk_col in enumerate(fk_columns):
                                ref_col = ref_columns[i] if i < len(ref_columns) else fk_col
                                facts.append(f"foreign_key({table_name}, {fk_col}, {ref_table}, {ref_col})")
        
        self.logger.debug(f"Generated {len(facts)} facts from schema")
        return facts
    
    def convert_to_facts(self, sql_query: str, schema: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Convert SQL query and schema to logical facts
        
        Args:
            sql_query: SQL query string
            schema: Optional database schema
            
        Returns:
            Combined list of logical facts
        """
        facts = []
        
        # Convert query to facts
        query_facts = self.convert_query_to_facts(sql_query)
        facts.extend(query_facts)
        
        # Convert schema to facts
        if schema:
            schema_facts = self.convert_schema_to_facts(schema)
            facts.extend(schema_facts)
        elif self.knowledge_base:
            schema_facts = self.knowledge_base.generate_facts()
            facts.extend(schema_facts)
        
        # Add interaction facts (query-schema relationships)
        analysis = self.analyze_query(sql_query)
        
        for table in analysis.tables:
            facts.append(f"query_accesses_table({table})")
        
        for table, column in analysis.columns:
            facts.append(f"query_accesses_column({table}, {column})")
        
        # Add validation facts
        facts.append(f"query_analyzed('{sql_query[:50]}...')")
        facts.append(f"facts_generated({len(facts)})")
        
        self.logger.info(f"Generated {len(facts)} total facts for query and schema")
        return facts
    
    def generate_constraint_facts(self, violations: List[Any]) -> List[str]:
        """
        Generate facts from constraint violations
        
        Args:
            violations: List of constraint violations
            
        Returns:
            List of violation facts
        """
        facts = []
        
        for i, violation in enumerate(violations):
            if hasattr(violation, 'violation_type'):
                violation_type = violation.violation_type.value
                table = getattr(violation, 'table', None)
                column = getattr(violation, 'column', None)
                
                fact_parts = [violation_type]
                if table:
                    fact_parts.append(table)
                if column:
                    fact_parts.append(column)
                
                facts.append(f"violation({', '.join(fact_parts)})")
            else:
                # Generic violation
                facts.append(f"violation(unknown_violation_{i})")
        
        return facts
    
    def format_facts_for_reasoning(self, facts: List[str]) -> str:
        """
        Format facts for symbolic reasoning engine
        
        Args:
            facts: List of logical facts
            
        Returns:
            Formatted facts string
        """
        formatted_facts = []
        
        for fact in facts:
            # Ensure proper fact format
            if not fact.endswith('.'):
                fact += '.'
            formatted_facts.append(fact)
        
        return '\n'.join(formatted_facts)
    
    def get_fact_statistics(self, facts: List[str]) -> Dict[str, Any]:
        """Get statistics about generated facts"""
        stats = {
            'total_facts': len(facts),
            'fact_types': {},
            'tables_referenced': set(),
            'columns_referenced': set()
        }
        
        for fact in facts:
            # Extract fact type (predicate name)
            if '(' in fact:
                fact_type = fact.split('(')[0]
                stats['fact_types'][fact_type] = stats['fact_types'].get(fact_type, 0) + 1
            
            # Extract table references
            if 'table' in fact and '(' in fact:
                # Simple extraction - could be more sophisticated
                match = re.search(r'(\w+)\(([^)]+)\)', fact)
                if match:
                    args = [arg.strip() for arg in match.group(2).split(',')]
                    if len(args) >= 1 and not args[0].startswith("'"):
                        stats['tables_referenced'].add(args[0])
                    if len(args) >= 2 and not args[1].startswith("'"):
                        stats['columns_referenced'].add(f"{args[0]}.{args[1]}")
        
        # Convert sets to lists for JSON serialization
        stats['tables_referenced'] = list(stats['tables_referenced'])
        stats['columns_referenced'] = list(stats['columns_referenced'])
        
        return stats


# Convenience functions
def create_converter(knowledge_base: Optional[SQLKnowledgeBase] = None) -> SQLToFactsConverter:
    """Create a new SQL to facts converter"""
    return SQLToFactsConverter(knowledge_base)


def convert_sql_to_facts(sql_query: str, schema: Optional[Dict[str, Any]] = None,
                        knowledge_base: Optional[SQLKnowledgeBase] = None) -> List[str]:
    """Convenience function to convert SQL to facts"""
    converter = SQLToFactsConverter(knowledge_base)
    return converter.convert_to_facts(sql_query, schema)