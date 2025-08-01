#!/usr/bin/env python3
"""
Simple Logic Engine

A custom constraint validation and reasoning engine designed to replace PyReason
for SQL constraint checking without the numba compilation issues.
"""

import logging
import re
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class ConstraintType(Enum):
    PRIMARY_KEY = "primary_key"
    FOREIGN_KEY = "foreign_key"
    NOT_NULL = "not_null"
    UNIQUE = "unique"
    CHECK = "check"
    DATA_TYPE = "data_type"


@dataclass(frozen=True)  # Make hashable for set operations
class LogicFact:
    """Represents a logical fact"""
    predicate: str
    arguments: tuple  # Use tuple instead of list for hashability
    confidence: float = 1.0
    
    def __str__(self):
        return f"{self.predicate}({', '.join(self.arguments)})"


@dataclass
class LogicRule:
    """Represents a logical rule (head :- body)"""
    head: LogicFact
    body: List[LogicFact]
    confidence: float = 1.0
    
    def __str__(self):
        body_str = ', '.join(str(fact) for fact in self.body)
        return f"{self.head} :- {body_str}"


@dataclass
class ValidationViolation:
    """Represents a constraint violation"""
    constraint_type: ConstraintType
    severity: str
    message: str
    affected_tables: List[str]
    affected_columns: List[str]
    confidence: float


class SimpleLogicEngine:
    """
    Simple logic engine for SQL constraint validation
    
    This replaces PyReason with a custom implementation that doesn't
    require numba compilation and is specifically designed for SQL
    constraint checking.
    """
    
    def __init__(self):
        self.facts: Set[LogicFact] = set()
        self.rules: List[LogicRule] = []
        self.violations: List[ValidationViolation] = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize built-in SQL constraint rules
        self._initialize_sql_rules()
    
    def _initialize_sql_rules(self):
        """Initialize built-in SQL constraint validation rules"""
        
        # Primary key violation rules
        self.add_rule_from_string(
            "violation(primary_key_duplicate) :- primary_key(Table, Column), duplicate_values(Table, Column)"
        )
        self.add_rule_from_string(
            "violation(primary_key_null) :- primary_key(Table, Column), null_value(Table, Column)"
        )
        
        # Foreign key violation rules
        self.add_rule_from_string(
            "violation(foreign_key_invalid) :- foreign_key(Table1, Column1, Table2, Column2), missing_reference(Table1, Column1, Table2, Column2)"
        )
        
        # Not null violation rules
        self.add_rule_from_string(
            "violation(not_null) :- not_null_constraint(Table, Column), null_value(Table, Column)"
        )
        
        # Unique constraint violation rules
        self.add_rule_from_string(
            "violation(unique_constraint) :- unique_constraint(Table, Column), duplicate_values(Table, Column)"
        )
        
        # Data type violation rules
        self.add_rule_from_string(
            "violation(type_mismatch) :- column_type(Table, Column, ExpectedType), value_type(Table, Column, ActualType), type_mismatch(ExpectedType, ActualType)"
        )
        
        self.logger.info(f"Initialized {len(self.rules)} built-in SQL constraint rules")
    
    def add_fact(self, fact: LogicFact):
        """Add a fact to the knowledge base"""
        self.facts.add(fact)
        self.logger.debug(f"Added fact: {fact}")
    
    def add_fact_from_string(self, fact_str: str, confidence: float = 1.0):
        """Add a fact from string format: predicate(arg1, arg2, ...)"""
        fact = self._parse_fact_string(fact_str, confidence)
        if fact:
            self.add_fact(fact)
    
    def add_rule(self, rule: LogicRule):
        """Add a rule to the knowledge base"""
        self.rules.append(rule)
        self.logger.debug(f"Added rule: {rule}")
    
    def add_rule_from_string(self, rule_str: str, confidence: float = 1.0):
        """Add a rule from string format: head :- body1, body2, ..."""
        rule = self._parse_rule_string(rule_str, confidence)
        if rule:
            self.add_rule(rule)
    
    def _parse_fact_string(self, fact_str: str, confidence: float = 1.0) -> Optional[LogicFact]:
        """Parse a fact string into a LogicFact object"""
        # Match pattern: predicate(arg1, arg2, ...)
        match = re.match(r'(\w+)\((.*)\)', fact_str.strip())
        if not match:
            self.logger.warning(f"Invalid fact format: {fact_str}")
            return None
        
        predicate = match.group(1)
        args_str = match.group(2).strip()
        
        if args_str:
            # Split arguments and clean them
            arguments = tuple(arg.strip().strip('"\'') for arg in args_str.split(','))
        else:
            arguments = tuple()
        
        return LogicFact(predicate, arguments, confidence)
    
    def _parse_rule_string(self, rule_str: str, confidence: float = 1.0) -> Optional[LogicRule]:
        """Parse a rule string into a LogicRule object"""
        # Clean the rule string and handle multi-line formatting
        clean_rule = ' '.join(rule_str.split())
        
        # Match pattern: head :- body1, body2, ...
        if ':-' not in clean_rule:
            self.logger.warning(f"Invalid rule format (missing ':-'): {clean_rule}")
            return None
        
        head_str, body_str = clean_rule.split(':-', 1)
        
        # Parse head
        head = self._parse_fact_string(head_str.strip(), confidence)
        if not head:
            return None
        
        # Parse body facts - handle special syntax like ~exists_reference and !=
        body_facts = []
        if body_str.strip():
            # Split on commas but be careful about commas inside predicates
            parts = []
            current_part = ""
            paren_count = 0
            
            for char in body_str:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                elif char == ',' and paren_count == 0:
                    parts.append(current_part.strip())
                    current_part = ""
                    continue
                current_part += char
            
            if current_part.strip():
                parts.append(current_part.strip())
            
            for part in parts:
                part = part.strip()
                # Skip special conditions for now (like != comparisons)
                if '!=' in part or '~' in part:
                    continue
                    
                body_fact = self._parse_fact_string(part, confidence)
                if body_fact:
                    body_facts.append(body_fact)
        
        return LogicRule(head, body_facts, confidence)
    
    def reason(self, max_iterations: int = 10) -> Dict[str, Any]:
        """
        Perform forward chaining reasoning to derive new facts and find violations
        
        Returns:
            Dictionary with reasoning results including derived facts and violations
        """
        self.logger.info(f"Starting reasoning with {len(self.facts)} facts and {len(self.rules)} rules")
        
        derived_facts = set()
        reasoning_trace = []
        iteration = 0
        
        # Forward chaining loop
        while iteration < max_iterations:
            new_facts_derived = False
            iteration += 1
            
            for rule in self.rules:
                # Check if rule conditions are satisfied
                if self._rule_applies(rule):
                    # Generate new fact from rule head
                    new_fact = self._instantiate_rule_head(rule)
                    
                    if new_fact and new_fact not in self.facts and new_fact not in derived_facts:
                        derived_facts.add(new_fact)
                        reasoning_trace.append(f"Iteration {iteration}: Derived {new_fact} from rule {rule}")
                        new_facts_derived = True
                        
                        # Check if this is a violation
                        if new_fact.predicate == "violation":
                            violation = self._create_violation_from_fact(new_fact)
                            if violation:
                                self.violations.append(violation)
            
            # Add derived facts to knowledge base for next iteration
            self.facts.update(derived_facts)
            
            # Stop if no new facts were derived
            if not new_facts_derived:
                break
        
        results = {
            'facts': list(self.facts),
            'derived_facts': list(derived_facts),
            'violations': self.violations,
            'trace': reasoning_trace,
            'iterations': iteration,
            'confidence': self._calculate_overall_confidence()
        }
        
        self.logger.info(f"Reasoning completed in {iteration} iterations: {len(derived_facts)} new facts, {len(self.violations)} violations")
        return results
    
    def _rule_applies(self, rule: LogicRule) -> bool:
        """Check if all conditions in rule body are satisfied by current facts"""
        for body_fact in rule.body:
            if not self._fact_matches_any(body_fact):
                return False
        return True
    
    def _fact_matches_any(self, target_fact: LogicFact) -> bool:
        """Check if target fact matches any fact in knowledge base (allowing variables)"""
        for kb_fact in self.facts:
            if self._facts_unify(target_fact, kb_fact):
                return True
        return False
    
    def _facts_unify(self, fact1: LogicFact, fact2: LogicFact) -> bool:
        """Check if two facts unify (simple pattern matching)"""
        if fact1.predicate != fact2.predicate:
            return False
        
        if len(fact1.arguments) != len(fact2.arguments):
            return False
        
        for arg1, arg2 in zip(fact1.arguments, fact2.arguments):
            # Variables (capitalized) match anything
            if arg1.isupper() or arg2.isupper():
                continue
            # Literals must match exactly
            if arg1 != arg2:
                return False
        
        return True
    
    def _instantiate_rule_head(self, rule: LogicRule) -> Optional[LogicFact]:
        """Instantiate rule head with concrete values from matching body facts"""
        # For simplicity, return the rule head as-is
        # In a full implementation, this would substitute variables
        return LogicFact(
            rule.head.predicate,
            rule.head.arguments,  # arguments is already a tuple
            rule.head.confidence
        )
    
    def _create_violation_from_fact(self, violation_fact: LogicFact) -> Optional[ValidationViolation]:
        """Create a ValidationViolation from a violation fact"""
        if violation_fact.predicate != "violation":
            return None
        
        if not violation_fact.arguments:
            return None
        
        violation_type = violation_fact.arguments[0]
        
        # Map violation types to constraint types
        constraint_type_map = {
            'primary_key_duplicate': ConstraintType.PRIMARY_KEY,
            'primary_key_null': ConstraintType.PRIMARY_KEY,
            'foreign_key_invalid': ConstraintType.FOREIGN_KEY,
            'not_null': ConstraintType.NOT_NULL,
            'unique_constraint': ConstraintType.UNIQUE,
            'type_mismatch': ConstraintType.DATA_TYPE
        }
        
        constraint_type = constraint_type_map.get(violation_type, ConstraintType.CHECK)
        
        return ValidationViolation(
            constraint_type=constraint_type,
            severity="error",
            message=f"Constraint violation: {violation_type}",
            affected_tables=violation_fact.arguments[1:2] if len(violation_fact.arguments) > 1 else [],
            affected_columns=violation_fact.arguments[2:3] if len(violation_fact.arguments) > 2 else [],
            confidence=violation_fact.confidence
        )
    
    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence of reasoning results"""
        if not self.facts:
            return 1.0
        
        total_confidence = sum(fact.confidence for fact in self.facts)
        return total_confidence / len(self.facts)
    
    def validate_sql_constraints(self, sql_facts: List[str], schema_facts: List[str]) -> Dict[str, Any]:
        """
        Validate SQL query against schema constraints
        
        Args:
            sql_facts: List of facts derived from SQL query
            schema_facts: List of facts derived from database schema
            
        Returns:
            Validation results with violations and confidence
        """
        # Clear previous state
        self.facts.clear()
        self.violations.clear()
        
        # Add SQL and schema facts
        for fact_str in sql_facts + schema_facts:
            self.add_fact_from_string(fact_str)
        
        # Run reasoning
        results = self.reason()
        
        return {
            'is_valid': len(self.violations) == 0,
            'violations': [v.message for v in self.violations],
            'detailed_violations': self.violations,
            'confidence': results['confidence'],
            'reasoning_trace': results['trace'],
            'facts_count': len(results['facts']),
            'iterations': results['iterations']
        }
    
    def reset(self):
        """Reset the engine state"""
        self.facts.clear()
        self.violations.clear()
        self._initialize_sql_rules()
        self.logger.debug("Logic engine reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            'facts_count': len(self.facts),
            'rules_count': len(self.rules),
            'violations_count': len(self.violations),
            'engine_type': 'SimpleLogicEngine'
        }


def create_simple_logic_engine() -> SimpleLogicEngine:
    """Factory function to create a SimpleLogicEngine instance"""
    return SimpleLogicEngine()


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create engine
    engine = SimpleLogicEngine()
    
    # Add some test facts
    engine.add_fact_from_string("table_exists(users)")
    engine.add_fact_from_string("column_exists(users, id)")
    engine.add_fact_from_string("column_exists(users, name)")
    engine.add_fact_from_string("primary_key(users, id)")
    engine.add_fact_from_string("not_null_constraint(users, name)")
    
    # Add a violation case
    engine.add_fact_from_string("null_value(users, name)")
    
    # Run reasoning
    results = engine.reason()
    
    print(f"Reasoning completed:")
    print(f"- Facts: {len(results['facts'])}")
    print(f"- Violations: {len(results['violations'])}")
    print(f"- Confidence: {results['confidence']:.3f}")
    
    if results['violations']:
        print("Violations found:")
        for violation in results['violations']:
            print(f"  - {violation}")