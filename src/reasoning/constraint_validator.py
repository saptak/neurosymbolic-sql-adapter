#!/usr/bin/env python3
"""
Constraint Validator

Validates SQL queries against database constraints using symbolic reasoning.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import sqlparse
from sqlparse.sql import Statement, Token, Identifier, IdentifierList, Function
from sqlparse.tokens import Keyword, Name

from .sql_knowledge_base import SQLKnowledgeBase, ConstraintType, DataType


class ViolationType(Enum):
    """Types of constraint violations"""
    PRIMARY_KEY_DUPLICATE = "primary_key_duplicate"
    PRIMARY_KEY_NULL = "primary_key_null"
    FOREIGN_KEY_INVALID = "foreign_key_invalid"
    NOT_NULL_VIOLATION = "not_null_violation"
    UNIQUE_VIOLATION = "unique_violation"
    TYPE_MISMATCH = "type_mismatch"
    TABLE_NOT_EXISTS = "table_not_exists"
    COLUMN_NOT_EXISTS = "column_not_exists"
    REFERENCE_ERROR = "reference_error"
    SYNTAX_ERROR = "syntax_error"


@dataclass
class Violation:
    """Represents a constraint violation"""
    violation_type: ViolationType
    severity: str = "error"  # error, warning, info
    table: Optional[str] = None
    column: Optional[str] = None
    message: str = ""
    sql_fragment: str = ""
    suggestion: Optional[str] = None
    
    def __str__(self) -> str:
        location = ""
        if self.table and self.column:
            location = f" in {self.table}.{self.column}"
        elif self.table:
            location = f" in table {self.table}"
        
        return f"{self.violation_type.value}{location}: {self.message}"


class SQLParser:
    """Helper class for parsing SQL queries"""
    
    @staticmethod
    def parse_query(sql: str) -> Statement:
        """Parse SQL query into AST"""
        parsed = sqlparse.parse(sql)
        if not parsed:
            raise ValueError("Could not parse SQL query")
        return parsed[0]
    
    @staticmethod
    def extract_tables(statement: Statement) -> List[str]:
        """Extract table names from SQL statement"""
        tables = []
        
        def extract_from_token(token):
            if token.ttype is Keyword and token.value.upper() == 'FROM':
                return True
            return False
        
        # Find FROM clause
        from_seen = False
        for token in statement.flatten():
            if token.ttype is Keyword and token.value.upper() == 'FROM':
                from_seen = True
                continue
            
            if from_seen:
                if token.ttype is Keyword:
                    break
                if token.ttype is Name:
                    tables.append(token.value)
        
        return tables
    
    @staticmethod
    def extract_columns(statement: Statement) -> List[Tuple[str, str]]:
        """Extract column references (table, column) from SQL statement"""
        columns = []
        
        for token in statement.flatten():
            if '.' in str(token) and token.ttype is Name:
                parts = str(token).split('.')
                if len(parts) == 2:
                    columns.append((parts[0], parts[1]))
        
        return columns
    
    @staticmethod
    def extract_select_columns(statement: Statement) -> List[str]:
        """Extract column names from SELECT clause"""
        columns = []
        select_seen = False
        
        for token in statement.flatten():
            if token.ttype is Keyword and token.value.upper() == 'SELECT':
                select_seen = True
                continue
            
            if select_seen:
                if token.ttype is Keyword and token.value.upper() == 'FROM':
                    break
                if token.ttype is Name:
                    columns.append(token.value)
        
        return columns


class ConstraintValidator:
    """
    Validates SQL queries against database constraints
    
    Uses the SQLKnowledgeBase to check for constraint violations
    in SQL queries before execution.
    """
    
    def __init__(self, knowledge_base: SQLKnowledgeBase):
        """
        Initialize constraint validator
        
        Args:
            knowledge_base: SQL knowledge base with schema information
        """
        self.logger = logging.getLogger(__name__)
        self.knowledge_base = knowledge_base
        self.parser = SQLParser()
        
        # Violation detectors mapping
        self.violation_detectors = {
            ViolationType.TABLE_NOT_EXISTS: self.check_table_existence,
            ViolationType.COLUMN_NOT_EXISTS: self.check_column_existence,
            ViolationType.PRIMARY_KEY_DUPLICATE: self.check_primary_key_uniqueness,
            ViolationType.FOREIGN_KEY_INVALID: self.check_referential_integrity,
            ViolationType.NOT_NULL_VIOLATION: self.check_not_null_constraints,
            ViolationType.UNIQUE_VIOLATION: self.check_unique_constraints,
            ViolationType.TYPE_MISMATCH: self.check_type_constraints
        }
        
        self.logger.info("Constraint validator initialized")
    
    def validate_query(self, sql_query: str) -> List[Violation]:
        """
        Validate SQL query against all constraints
        
        Args:
            sql_query: SQL query string to validate
            
        Returns:
            List of constraint violations found
        """
        violations = []
        
        try:
            # Parse SQL query
            statement = self.parser.parse_query(sql_query)
            
            # Run all violation detectors
            for violation_type, detector in self.violation_detectors.items():
                try:
                    detected_violations = detector(statement, sql_query)
                    violations.extend(detected_violations)
                except Exception as e:
                    self.logger.error(f"Error in {violation_type.value} detector: {e}")
                    violations.append(Violation(
                        violation_type=ViolationType.SYNTAX_ERROR,
                        message=f"Error checking {violation_type.value}: {str(e)}",
                        sql_fragment=sql_query[:100]
                    ))
        
        except Exception as e:
            self.logger.error(f"Error parsing SQL query: {e}")
            violations.append(Violation(
                violation_type=ViolationType.SYNTAX_ERROR,
                message=f"SQL parsing error: {str(e)}",
                sql_fragment=sql_query[:100]
            ))
        
        self.logger.debug(f"Found {len(violations)} violations in query validation")
        return violations
    
    def check_table_existence(self, statement: Statement, sql_query: str) -> List[Violation]:
        """Check if all referenced tables exist"""
        violations = []
        tables = self.parser.extract_tables(statement)
        
        for table in tables:
            if not self.knowledge_base.validate_table_reference(table):
                violations.append(Violation(
                    violation_type=ViolationType.TABLE_NOT_EXISTS,
                    table=table,
                    message=f"Table '{table}' does not exist",
                    sql_fragment=sql_query,
                    suggestion=f"Check table name spelling or create table '{table}'"
                ))
        
        return violations
    
    def check_column_existence(self, statement: Statement, sql_query: str) -> List[Violation]:
        """Check if all referenced columns exist"""
        violations = []
        
        # Extract table.column references
        column_refs = self.parser.extract_columns(statement)
        tables = self.parser.extract_tables(statement)
        
        for table, column in column_refs:
            if not self.knowledge_base.validate_column_reference(table, column):
                violations.append(Violation(
                    violation_type=ViolationType.COLUMN_NOT_EXISTS,
                    table=table,
                    column=column,
                    message=f"Column '{column}' does not exist in table '{table}'",
                    sql_fragment=sql_query,
                    suggestion=f"Check column name spelling in table '{table}'"
                ))
        
        # Check SELECT columns against available tables
        select_columns = self.parser.extract_select_columns(statement)
        
        for column in select_columns:
            if column == '*':
                continue
            
            # Check if column exists in any referenced table
            column_found = False
            for table in tables:
                if self.knowledge_base.validate_column_reference(table, column):
                    column_found = True
                    break
            
            if not column_found and tables:
                violations.append(Violation(
                    violation_type=ViolationType.COLUMN_NOT_EXISTS,
                    column=column,
                    message=f"Column '{column}' not found in any referenced table",
                    sql_fragment=sql_query,
                    suggestion=f"Specify table name as table.{column} or check column name"
                ))
        
        return violations
    
    def check_primary_key_uniqueness(self, statement: Statement, sql_query: str) -> List[Violation]:
        """Check primary key uniqueness constraints"""
        violations = []
        
        # This is a static analysis - we can't check actual duplicates without data
        # But we can check if INSERT statements properly handle primary keys
        
        sql_upper = sql_query.upper()
        if 'INSERT' in sql_upper:
            tables = self.parser.extract_tables(statement)
            
            for table in tables:
                pk_columns = self.knowledge_base.get_primary_key_columns(table)
                
                if pk_columns:
                    # Check if INSERT specifies primary key columns
                    # This is a simplified check - real implementation would be more complex
                    for pk_column in pk_columns:
                        if pk_column.lower() not in sql_query.lower():
                            violations.append(Violation(
                                violation_type=ViolationType.PRIMARY_KEY_NULL,
                                table=table,
                                column=pk_column,
                                message=f"Primary key column '{pk_column}' should be specified in INSERT",
                                sql_fragment=sql_query,
                                severity="warning",
                                suggestion=f"Include {pk_column} in INSERT statement"
                            ))
        
        return violations
    
    def check_referential_integrity(self, statement: Statement, sql_query: str) -> List[Violation]:
        """Check foreign key referential integrity"""
        violations = []
        
        tables = self.parser.extract_tables(statement)
        
        for table in tables:
            fk_relationships = self.knowledge_base.get_foreign_key_relationships(table)
            
            for relationship in fk_relationships:
                # Check if referenced table exists
                if relationship.from_table == table:
                    if not self.knowledge_base.validate_table_reference(relationship.to_table):
                        violations.append(Violation(
                            violation_type=ViolationType.FOREIGN_KEY_INVALID,
                            table=table,
                            column=relationship.from_column,
                            message=f"Foreign key references non-existent table '{relationship.to_table}'",
                            sql_fragment=sql_query,
                            suggestion=f"Ensure table '{relationship.to_table}' exists"
                        ))
                    
                    # Check if referenced column exists
                    if not self.knowledge_base.validate_column_reference(relationship.to_table, relationship.to_column):
                        violations.append(Violation(
                            violation_type=ViolationType.FOREIGN_KEY_INVALID,
                            table=table,
                            column=relationship.from_column,
                            message=f"Foreign key references non-existent column '{relationship.to_table}.{relationship.to_column}'",
                            sql_fragment=sql_query,
                            suggestion=f"Check column name in referenced table"
                        ))
        
        return violations
    
    def check_not_null_constraints(self, statement: Statement, sql_query: str) -> List[Violation]:
        """Check NOT NULL constraints"""
        violations = []
        
        sql_upper = sql_query.upper()
        if 'INSERT' in sql_upper or 'UPDATE' in sql_upper:
            tables = self.parser.extract_tables(statement)
            
            for table in tables:
                table_obj = self.knowledge_base.get_table(table)
                if not table_obj:
                    continue
                
                # Check columns with NOT NULL constraints
                for column_name, column in table_obj.columns.items():
                    if not column.nullable:
                        # This is a simplified check - real implementation would parse VALUES clause
                        if 'NULL' in sql_upper and column_name.upper() in sql_upper:
                            violations.append(Violation(
                                violation_type=ViolationType.NOT_NULL_VIOLATION,
                                table=table,
                                column=column_name,
                                message=f"Column '{column_name}' cannot be NULL",
                                sql_fragment=sql_query,
                                suggestion=f"Provide a value for '{column_name}'"
                            ))
        
        return violations
    
    def check_unique_constraints(self, statement: Statement, sql_query: str) -> List[Violation]:
        """Check UNIQUE constraints"""
        violations = []
        
        # Check for unique constraints in knowledge base
        for constraint_id, constraint in self.knowledge_base.constraints.items():
            if constraint['type'] == ConstraintType.UNIQUE.value:
                table = constraint['table']
                columns = constraint['columns']
                
                # This is a static check - would need runtime data for full validation
                if table in sql_query.lower():
                    for column in columns:
                        if column in sql_query.lower():
                            violations.append(Violation(
                                violation_type=ViolationType.UNIQUE_VIOLATION,
                                table=table,
                                column=column,
                                message=f"Column '{column}' has UNIQUE constraint",
                                sql_fragment=sql_query,
                                severity="info",
                                suggestion=f"Ensure unique values for '{column}'"
                            ))
        
        return violations
    
    def check_type_constraints(self, statement: Statement, sql_query: str) -> List[Violation]:
        """Check data type constraints"""
        violations = []
        
        # Extract literal values and check against column types
        # This is a simplified implementation
        
        tables = self.parser.extract_tables(statement)
        for table in tables:
            table_obj = self.knowledge_base.get_table(table)
            if not table_obj:
                continue
            
            # Check for obvious type mismatches in literals
            for column_name, column in table_obj.columns.items():
                if column_name.lower() in sql_query.lower():
                    # Simple heuristic checks
                    if column.data_type == DataType.INTEGER:
                        # Look for non-numeric values
                        if re.search(rf"{column_name}\s*=\s*'[a-zA-Z]", sql_query, re.IGNORECASE):
                            violations.append(Violation(
                                violation_type=ViolationType.TYPE_MISMATCH,
                                table=table,
                                column=column_name,
                                message=f"Expected INTEGER for column '{column_name}'",
                                sql_fragment=sql_query,
                                severity="warning",
                                suggestion=f"Use numeric value for '{column_name}'"
                            ))
                    
                    elif column.data_type == DataType.DATE:
                        # Look for invalid date formats
                        if re.search(rf"{column_name}\s*=\s*'\d{{4}}-\d{{2}}-\d{{2}}'", sql_query, re.IGNORECASE):
                            pass  # Valid date format
                        elif re.search(rf"{column_name}\s*=\s*'[^']+'", sql_query, re.IGNORECASE):
                            violations.append(Violation(
                                violation_type=ViolationType.TYPE_MISMATCH,
                                table=table,
                                column=column_name,
                                message=f"Expected DATE format YYYY-MM-DD for column '{column_name}'",
                                sql_fragment=sql_query,
                                severity="warning",
                                suggestion=f"Use proper date format for '{column_name}'"
                            ))
        
        return violations
    
    def validate_facts(self, facts: List[str]) -> List[Violation]:
        """
        Validate logical facts against constraints
        
        Args:
            facts: List of logical facts to validate
            
        Returns:
            List of violations found
        """
        violations = []
        
        # Parse facts and check against constraints
        for fact in facts:
            if 'violation(' in fact:
                # Extract violation information from fact
                match = re.match(r'violation\((\w+),\s*([^)]+)\)', fact)
                if match:
                    violation_type_str = match.group(1)
                    parameters = match.group(2).split(',')
                    
                    try:
                        violation_type = ViolationType(violation_type_str)
                        
                        violation = Violation(
                            violation_type=violation_type,
                            message=f"Constraint violation detected: {fact}",
                            sql_fragment=fact
                        )
                        
                        # Extract table and column if available
                        if len(parameters) >= 2:
                            violation.table = parameters[0].strip()
                            violation.column = parameters[1].strip()
                        
                        violations.append(violation)
                        
                    except ValueError:
                        # Unknown violation type
                        violations.append(Violation(
                            violation_type=ViolationType.REFERENCE_ERROR,
                            message=f"Unknown violation type: {violation_type_str}",
                            sql_fragment=fact
                        ))
        
        return violations
    
    def get_violation_summary(self, violations: List[Violation]) -> Dict[str, Any]:
        """Get summary of violations by type and severity"""
        summary = {
            'total_violations': len(violations),
            'by_type': {},
            'by_severity': {'error': 0, 'warning': 0, 'info': 0},
            'by_table': {}
        }
        
        for violation in violations:
            # Count by type
            type_name = violation.violation_type.value
            summary['by_type'][type_name] = summary['by_type'].get(type_name, 0) + 1
            
            # Count by severity
            summary['by_severity'][violation.severity] += 1
            
            # Count by table
            if violation.table:
                summary['by_table'][violation.table] = summary['by_table'].get(violation.table, 0) + 1
        
        return summary
    
    def format_violations_report(self, violations: List[Violation]) -> str:
        """Format violations into a readable report"""
        if not violations:
            return "✅ No constraint violations found."
        
        report = []
        report.append(f"🚨 Found {len(violations)} constraint violations:\n")
        
        # Group by severity
        errors = [v for v in violations if v.severity == 'error']
        warnings = [v for v in violations if v.severity == 'warning']
        infos = [v for v in violations if v.severity == 'info']
        
        for severity, viols in [('ERRORS', errors), ('WARNINGS', warnings), ('INFO', infos)]:
            if viols:
                report.append(f"{severity}:")
                for i, violation in enumerate(viols, 1):
                    report.append(f"  {i}. {violation}")
                    if violation.suggestion:
                        report.append(f"     💡 {violation.suggestion}")
                report.append("")
        
        summary = self.get_violation_summary(violations)
        report.append("📊 Summary:")
        report.append(f"   Total: {summary['total_violations']}")
        report.append(f"   Errors: {summary['by_severity']['error']}")
        report.append(f"   Warnings: {summary['by_severity']['warning']}")
        report.append(f"   Info: {summary['by_severity']['info']}")
        
        return '\n'.join(report)


# Convenience functions
def create_validator(knowledge_base: SQLKnowledgeBase) -> ConstraintValidator:
    """Create a new constraint validator"""
    return ConstraintValidator(knowledge_base)


def validate_sql_query(sql_query: str, knowledge_base: SQLKnowledgeBase) -> List[Violation]:
    """Convenience function to validate a single SQL query"""
    validator = ConstraintValidator(knowledge_base)
    return validator.validate_query(sql_query)