#!/usr/bin/env python3
"""
Configuration Loader

Loads and manages SQL reasoning rules and configuration settings.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ReasoningRule:
    """Represents a single reasoning rule"""
    rule_id: str
    name: str
    description: str
    logic: str
    pyreason_format: Dict[str, Any]
    priority: str = "medium"


@dataclass
class ReasoningConfiguration:
    """Main reasoning configuration"""
    max_reasoning_steps: int
    inconsistency_tolerance: float
    explanation_depth: int
    confidence_threshold: float
    enable_optimizations: bool
    enable_warnings: bool
    parallel_reasoning: bool
    cache_intermediate_results: bool
    violation_severity_levels: Dict[str, List[str]]


class ConfigurationLoader:
    """
    Loads and manages SQL reasoning rules and configuration
    
    Provides access to reasoning rules, constraint definitions,
    and configuration settings for the neurosymbolic reasoning engine.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.logger = logging.getLogger(__name__)
        
        # Default configuration path
        if config_path is None:
            current_dir = Path(__file__).parent.parent.parent
            config_path = current_dir / "configs" / "sql_reasoning_rules.json"
        
        self.config_path = Path(config_path)
        self.config_data = None
        self.reasoning_rules = []
        self.reasoning_config = None
        
        # Load configuration
        self.load_configuration()
        
        self.logger.info(f"Configuration loaded from {self.config_path}")
    
    def load_configuration(self) -> None:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = json.load(f)
            
            # Parse reasoning rules
            self._parse_reasoning_rules()
            
            # Parse reasoning configuration
            self._parse_reasoning_configuration()
            
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in configuration file: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
    
    def _parse_reasoning_rules(self) -> None:
        """Parse reasoning rules from configuration"""
        rules_data = self.config_data.get('reasoning_rules', [])
        rule_priorities = self.config_data.get('rule_priorities', {})
        
        # Create priority lookup
        priority_lookup = {}
        for priority, rule_ids in rule_priorities.items():
            for rule_id in rule_ids:
                priority_lookup[rule_id] = priority.replace('_rules', '')
        
        self.reasoning_rules = []
        for rule_data in rules_data:
            rule = ReasoningRule(
                rule_id=rule_data['rule_id'],
                name=rule_data['name'],
                description=rule_data['description'],
                logic=rule_data['logic'],
                pyreason_format=rule_data['pyreason_format'],
                priority=priority_lookup.get(rule_data['rule_id'], 'medium')
            )
            self.reasoning_rules.append(rule)
    
    def _parse_reasoning_configuration(self) -> None:
        """Parse reasoning configuration settings"""
        config_data = self.config_data.get('reasoning_configuration', {})
        
        self.reasoning_config = ReasoningConfiguration(
            max_reasoning_steps=config_data.get('max_reasoning_steps', 10),
            inconsistency_tolerance=config_data.get('inconsistency_tolerance', 0.1),
            explanation_depth=config_data.get('explanation_depth', 3),
            confidence_threshold=config_data.get('confidence_threshold', 0.8),
            enable_optimizations=config_data.get('enable_optimizations', True),
            enable_warnings=config_data.get('enable_warnings', True),
            parallel_reasoning=config_data.get('parallel_reasoning', False),
            cache_intermediate_results=config_data.get('cache_intermediate_results', True),
            violation_severity_levels=config_data.get('violation_severity_levels', {})
        )
    
    def get_reasoning_rules(self, priority: Optional[str] = None) -> List[ReasoningRule]:
        """
        Get reasoning rules, optionally filtered by priority
        
        Args:
            priority: Filter by priority level (critical, high, medium, low)
            
        Returns:
            List of reasoning rules
        """
        if priority is None:
            return self.reasoning_rules
        
        return [rule for rule in self.reasoning_rules if rule.priority == priority]
    
    def get_rule_by_id(self, rule_id: str) -> Optional[ReasoningRule]:
        """
        Get a specific rule by ID
        
        Args:
            rule_id: Rule identifier
            
        Returns:
            ReasoningRule if found, None otherwise
        """
        for rule in self.reasoning_rules:
            if rule.rule_id == rule_id:
                return rule
        return None
    
    def get_constraint_definitions(self) -> Dict[str, Any]:
        """Get constraint definitions"""
        return self.config_data.get('constraint_definitions', {})
    
    def get_data_types(self) -> List[str]:
        """Get supported data types"""
        constraint_defs = self.get_constraint_definitions()
        return constraint_defs.get('data_types', [])
    
    def get_type_compatibility(self) -> Dict[str, List[str]]:
        """Get type compatibility groups"""
        constraint_defs = self.get_constraint_definitions()
        return constraint_defs.get('type_compatibility', {})
    
    def get_aggregate_functions(self) -> Dict[str, List[str]]:
        """Get aggregate functions and compatible types"""
        constraint_defs = self.get_constraint_definitions()
        return constraint_defs.get('aggregate_functions', {})
    
    def get_comparison_operators(self) -> Dict[str, List[str]]:
        """Get comparison operators and compatible types"""
        constraint_defs = self.get_constraint_definitions()
        return constraint_defs.get('comparison_operators', {})
    
    def get_arithmetic_operators(self) -> Dict[str, List[str]]:
        """Get arithmetic operators and compatible types"""
        constraint_defs = self.get_constraint_definitions()
        return constraint_defs.get('arithmetic_operators', {})
    
    def get_reasoning_configuration(self) -> ReasoningConfiguration:
        """Get reasoning configuration"""
        return self.reasoning_config
    
    def get_fact_templates(self) -> Dict[str, List[str]]:
        """Get fact templates"""
        return self.config_data.get('fact_templates', {})
    
    def get_explanation_templates(self) -> Dict[str, Dict[str, str]]:
        """Get explanation templates"""
        return self.config_data.get('explanation_templates', {})
    
    def get_violation_explanation(self, violation_type: str) -> Optional[str]:
        """
        Get explanation template for a violation type
        
        Args:
            violation_type: Type of violation
            
        Returns:
            Explanation template string if found
        """
        templates = self.get_explanation_templates()
        violation_templates = templates.get('violation_explanations', {})
        return violation_templates.get(violation_type)
    
    def get_optimization_suggestion(self, suggestion_type: str) -> Optional[str]:
        """
        Get optimization suggestion template
        
        Args:
            suggestion_type: Type of optimization
            
        Returns:
            Suggestion template string if found
        """
        templates = self.get_explanation_templates()
        optimization_templates = templates.get('optimization_suggestions', {})
        return optimization_templates.get(suggestion_type)
    
    def get_performance_settings(self) -> Dict[str, Any]:
        """Get performance settings"""
        return self.config_data.get('performance_settings', {})
    
    def is_type_compatible(self, type1: str, type2: str) -> bool:
        """
        Check if two data types are compatible
        
        Args:
            type1: First data type
            type2: Second data type
            
        Returns:
            True if types are compatible
        """
        if type1 == type2:
            return True
        
        type_compatibility = self.get_type_compatibility()
        
        # Check if both types are in the same compatibility group
        for group_types in type_compatibility.values():
            if type1 in group_types and type2 in group_types:
                return True
        
        return False
    
    def is_function_compatible(self, function: str, data_type: str) -> bool:
        """
        Check if a function is compatible with a data type
        
        Args:
            function: Function name
            data_type: Data type
            
        Returns:
            True if function is compatible with type
        """
        aggregate_functions = self.get_aggregate_functions()
        
        if function not in aggregate_functions:
            return False
        
        compatible_types = aggregate_functions[function]
        
        if "all_types" in compatible_types:
            return True
        
        return data_type in compatible_types
    
    def get_severity_level(self, violation_type: str) -> str:
        """
        Get severity level for a violation type
        
        Args:
            violation_type: Type of violation
            
        Returns:
            Severity level (critical, error, warning, info)
        """
        severity_levels = self.reasoning_config.violation_severity_levels
        
        for level, violation_types in severity_levels.items():
            if violation_type in violation_types:
                return level
        
        return "warning"  # Default severity
    
    def export_pyreason_rules(self) -> List[str]:
        """
        Export rules in PyReason format
        
        Returns:
            List of PyReason rule strings
        """
        pyreason_rules = []
        
        for rule in self.reasoning_rules:
            pyreason_format = rule.pyreason_format
            head = pyreason_format.get('head', '')
            body_conditions = pyreason_format.get('body', [])
            
            if head and body_conditions:
                body_str = ', '.join(body_conditions)
                rule_str = f"{head} :- {body_str}"
                pyreason_rules.append(rule_str)
        
        return pyreason_rules
    
    def validate_configuration(self) -> List[str]:
        """
        Validate configuration for completeness and consistency
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required sections
        required_sections = [
            'reasoning_rules',
            'constraint_definitions',
            'reasoning_configuration'
        ]
        
        for section in required_sections:
            if section not in self.config_data:
                errors.append(f"Missing required section: {section}")
        
        # Validate reasoning rules
        if not self.reasoning_rules:
            errors.append("No reasoning rules defined")
        
        for rule in self.reasoning_rules:
            if not rule.rule_id:
                errors.append("Rule missing ID")
            if not rule.pyreason_format:
                errors.append(f"Rule {rule.rule_id} missing PyReason format")
        
        # Validate reasoning configuration
        if self.reasoning_config:
            if self.reasoning_config.max_reasoning_steps <= 0:
                errors.append("max_reasoning_steps must be positive")
            if not 0 <= self.reasoning_config.confidence_threshold <= 1:
                errors.append("confidence_threshold must be between 0 and 1")
        
        return errors
    
    def reload_configuration(self) -> None:
        """Reload configuration from file"""
        self.load_configuration()
        self.logger.info("Configuration reloaded")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get configuration statistics"""
        return {
            'total_rules': len(self.reasoning_rules),
            'rules_by_priority': {
                priority: len([r for r in self.reasoning_rules if r.priority == priority])
                for priority in ['critical', 'high', 'medium', 'low']
            },
            'data_types_count': len(self.get_data_types()),
            'aggregate_functions_count': len(self.get_aggregate_functions()),
            'configuration_file': str(self.config_path),
            'max_reasoning_steps': self.reasoning_config.max_reasoning_steps,
            'optimizations_enabled': self.reasoning_config.enable_optimizations
        }


# Convenience functions
def load_default_configuration() -> ConfigurationLoader:
    """Load default configuration"""
    return ConfigurationLoader()


def load_configuration(config_path: str) -> ConfigurationLoader:
    """Load configuration from specific path"""
    return ConfigurationLoader(config_path)