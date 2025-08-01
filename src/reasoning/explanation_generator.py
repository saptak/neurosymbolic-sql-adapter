#!/usr/bin/env python3
"""
Explanation Generator

Generates human-readable explanations for symbolic reasoning results
and SQL query validation outcomes.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

from .constraint_validator import Violation, ViolationType
from .sql_to_facts import QueryAnalysis, QueryType


class ExplanationLevel(Enum):
    """Levels of explanation detail"""
    MINIMAL = "minimal"      # Just the outcome
    SUMMARY = "summary"      # Brief explanation
    DETAILED = "detailed"    # Comprehensive explanation
    TECHNICAL = "technical"  # Full technical details


class ExplanationStyle(Enum):
    """Styles of explanation presentation"""
    NATURAL = "natural"      # Natural language
    STRUCTURED = "structured"  # Bullet points/lists
    TECHNICAL = "technical"  # Technical format
    INTERACTIVE = "interactive"  # Q&A style


@dataclass
class ExplanationContext:
    """Context information for generating explanations"""
    query: str
    query_analysis: Optional[QueryAnalysis] = None
    violations: Optional[List[Violation]] = None
    facts: Optional[List[str]] = None
    reasoning_trace: Optional[List[str]] = None
    confidence: Optional[float] = None
    schema_info: Optional[Dict[str, Any]] = None
    user_level: str = "intermediate"  # beginner, intermediate, expert


@dataclass
class Explanation:
    """Generated explanation"""
    title: str
    summary: str
    details: List[str]
    recommendations: List[str]
    confidence_explanation: Optional[str] = None
    technical_details: Optional[Dict[str, Any]] = None
    references: Optional[List[str]] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ExplanationGenerator:
    """
    Generates human-readable explanations for symbolic reasoning results
    
    Provides comprehensive explanations of SQL validation results,
    constraint violations, and reasoning processes.
    """
    
    def __init__(self):
        """Initialize explanation generator"""
        self.logger = logging.getLogger(__name__)
        
        # Templates for different types of explanations
        self.templates = {
            'validation_success': {
                'title': "✅ SQL Query Validation Successful",
                'summary': "Your SQL query passed all validation checks.",
                'details_intro': "The query was validated against:"
            },
            'validation_failure': {
                'title': "❌ SQL Query Validation Failed",
                'summary': "Your SQL query has {violation_count} constraint violation(s).",
                'details_intro': "The following issues were found:"
            },
            'constraint_violation': {
                'table_not_exists': "Table '{table}' does not exist in the database schema",
                'column_not_exists': "Column '{column}' does not exist in table '{table}'",
                'primary_key_violation': "Primary key constraint violation in table '{table}'",
                'foreign_key_violation': "Foreign key constraint violation: {details}",
                'not_null_violation': "NOT NULL constraint violation for column '{table}.{column}'",
                'type_mismatch': "Data type mismatch for column '{table}.{column}': expected {expected}, got {actual}"
            }
        }
        
        # User level adaptations
        self.level_adaptations = {
            'beginner': {
                'use_technical_terms': False,
                'provide_examples': True,
                'explain_concepts': True,
                'detail_level': ExplanationLevel.SUMMARY
            },
            'intermediate': {
                'use_technical_terms': True,
                'provide_examples': True,
                'explain_concepts': False,
                'detail_level': ExplanationLevel.DETAILED
            },
            'expert': {
                'use_technical_terms': True,
                'provide_examples': False,
                'explain_concepts': False,
                'detail_level': ExplanationLevel.TECHNICAL
            }
        }
        
        self.logger.info("Explanation generator initialized")
    
    def generate_explanation(self, context: ExplanationContext,
                           level: ExplanationLevel = ExplanationLevel.DETAILED,
                           style: ExplanationStyle = ExplanationStyle.NATURAL) -> Explanation:
        """
        Generate explanation from context
        
        Args:
            context: Context information for explanation
            level: Level of detail to include
            style: Presentation style
            
        Returns:
            Generated explanation
        """
        user_prefs = self.level_adaptations.get(context.user_level, self.level_adaptations['intermediate'])
        
        # Determine if validation passed or failed
        has_violations = context.violations and len(context.violations) > 0
        
        if has_violations:
            return self._generate_failure_explanation(context, level, style, user_prefs)
        else:
            return self._generate_success_explanation(context, level, style, user_prefs)
    
    def _generate_success_explanation(self, context: ExplanationContext,
                                    level: ExplanationLevel,
                                    style: ExplanationStyle,
                                    user_prefs: Dict[str, Any]) -> Explanation:
        """Generate explanation for successful validation"""
        template = self.templates['validation_success']
        
        title = template['title']
        summary = template['summary']
        
        details = []
        recommendations = []
        
        # Add validation details
        if level in [ExplanationLevel.DETAILED, ExplanationLevel.TECHNICAL]:
            details.append(template['details_intro'])
            
            # Schema validation
            if context.schema_info:
                table_count = len(context.schema_info.get('tables', {}))
                details.append(f"• Database schema with {table_count} tables")
            
            # Query analysis
            if context.query_analysis:
                details.append(f"• Query type: {context.query_analysis.query_type.value}")
                details.append(f"• Tables referenced: {', '.join(context.query_analysis.tables)}")
                if context.query_analysis.joins:
                    details.append(f"• Join operations: {len(context.query_analysis.joins)}")
            
            # Constraint checking
            if context.facts:
                facts_count = len(context.facts)
                details.append(f"• {facts_count} logical facts analyzed")
            
            # Reasoning process
            if context.reasoning_trace:
                details.append(f"• {len(context.reasoning_trace)} reasoning steps executed")
        
        # Add confidence explanation
        confidence_explanation = None
        if context.confidence is not None:
            confidence_pct = int(context.confidence * 100)
            confidence_explanation = f"Validation confidence: {confidence_pct}%"
            
            if user_prefs['explain_concepts'] and context.confidence < 0.9:
                confidence_explanation += f" (Lower confidence may indicate complex query patterns or incomplete schema information)"
        
        # Add recommendations for successful queries
        if user_prefs['provide_examples']:
            recommendations.append("Consider testing the query with sample data before running in production")
            recommendations.append("Ensure proper indexing on columns used in WHERE and JOIN clauses")
        
        # Technical details
        technical_details = None
        if level == ExplanationLevel.TECHNICAL:
            technical_details = {
                'query_analysis': context.query_analysis.__dict__ if context.query_analysis else None,
                'facts_generated': len(context.facts) if context.facts else 0,
                'reasoning_steps': len(context.reasoning_trace) if context.reasoning_trace else 0,
                'validation_timestamp': datetime.now().isoformat()
            }
        
        return Explanation(
            title=title,
            summary=summary,
            details=details,
            recommendations=recommendations,
            confidence_explanation=confidence_explanation,
            technical_details=technical_details
        )
    
    def _generate_failure_explanation(self, context: ExplanationContext,
                                    level: ExplanationLevel,
                                    style: ExplanationStyle,
                                    user_prefs: Dict[str, Any]) -> Explanation:
        """Generate explanation for failed validation"""
        template = self.templates['validation_failure']
        violation_count = len(context.violations) if context.violations else 0
        
        title = template['title']
        summary = template['summary'].format(violation_count=violation_count)
        
        details = []
        recommendations = []
        
        # Group violations by type and severity
        if context.violations:
            violations_by_severity = {'error': [], 'warning': [], 'info': []}
            for violation in context.violations:
                violations_by_severity[violation.severity].append(violation)
            
            # Add violation details
            details.append(template['details_intro'])
            
            for severity in ['error', 'warning', 'info']:
                viols = violations_by_severity[severity]
                if viols:
                    severity_icon = {'error': '🚨', 'warning': '⚠️', 'info': 'ℹ️'}[severity]
                    details.append(f"\n{severity_icon} {severity.upper()}S ({len(viols)}):")
                    
                    for i, violation in enumerate(viols, 1):
                        violation_detail = self._format_violation(violation, user_prefs)
                        details.append(f"  {i}. {violation_detail}")
                        
                        # Add specific recommendations
                        if violation.suggestion:
                            recommendations.append(f"💡 {violation.suggestion}")
        
        # Add general recommendations
        if user_prefs['provide_examples']:
            if any(v.violation_type == ViolationType.TABLE_NOT_EXISTS for v in context.violations or []):
                recommendations.append("📚 Check the database schema documentation for correct table names")
            
            if any(v.violation_type == ViolationType.COLUMN_NOT_EXISTS for v in context.violations or []):
                recommendations.append("🔍 Use DESCRIBE or SHOW COLUMNS to verify column names")
        
        # Add reasoning explanation
        if level in [ExplanationLevel.DETAILED, ExplanationLevel.TECHNICAL] and context.reasoning_trace:
            details.append(f"\n🔬 REASONING PROCESS:")
            if user_prefs['explain_concepts']:
                details.append("The system analyzed your query using logical reasoning:")
            
            for i, step in enumerate(context.reasoning_trace[:5], 1):  # Show first 5 steps
                details.append(f"  {i}. {step}")
            
            if len(context.reasoning_trace) > 5:
                details.append(f"  ... and {len(context.reasoning_trace) - 5} more steps")
        
        # Confidence explanation
        confidence_explanation = None
        if context.confidence is not None:
            confidence_pct = int(context.confidence * 100)
            confidence_explanation = f"Validation confidence: {confidence_pct}%"
            
            if user_prefs['explain_concepts']:
                if context.confidence < 0.5:
                    confidence_explanation += " (Low confidence indicates significant issues)"
                elif context.confidence < 0.8:
                    confidence_explanation += " (Medium confidence suggests some uncertainty)"
        
        # Technical details
        technical_details = None
        if level == ExplanationLevel.TECHNICAL:
            technical_details = {
                'violations_by_type': {},
                'query_analysis': context.query_analysis.__dict__ if context.query_analysis else None,
                'facts_generated': len(context.facts) if context.facts else 0,
                'reasoning_steps': len(context.reasoning_trace) if context.reasoning_trace else 0
            }
            
            # Count violations by type
            if context.violations:
                for violation in context.violations:
                    vtype = violation.violation_type.value
                    technical_details['violations_by_type'][vtype] = \
                        technical_details['violations_by_type'].get(vtype, 0) + 1
        
        return Explanation(
            title=title,
            summary=summary,
            details=details,
            recommendations=recommendations,
            confidence_explanation=confidence_explanation,
            technical_details=technical_details
        )
    
    def _format_violation(self, violation: Violation, user_prefs: Dict[str, Any]) -> str:
        """Format a single violation for display"""
        violation_templates = self.templates['constraint_violation']
        
        # Get base message
        base_message = str(violation)
        
        # Enhance with template if available
        vtype = violation.violation_type.value
        if vtype in violation_templates:
            template = violation_templates[vtype]
            
            # Fill in template variables
            format_vars = {}
            if violation.table:
                format_vars['table'] = violation.table
            if violation.column:
                format_vars['column'] = violation.column
            
            try:
                formatted_message = template.format(**format_vars)
            except KeyError:
                formatted_message = base_message
        else:
            formatted_message = base_message
        
        # Add technical details for expert users
        if user_prefs['use_technical_terms'] and violation.sql_fragment:
            formatted_message += f" (in: {violation.sql_fragment[:50]}...)"
        
        return formatted_message
    
    def generate_reasoning_explanation(self, reasoning_trace: List[str],
                                     facts: List[str],
                                     level: ExplanationLevel = ExplanationLevel.DETAILED) -> str:
        """Generate explanation of reasoning process"""
        explanation_parts = []
        
        explanation_parts.append("🧠 REASONING PROCESS EXPLANATION:")
        explanation_parts.append("")
        
        if level == ExplanationLevel.MINIMAL:
            explanation_parts.append(f"Applied {len(reasoning_trace)} reasoning rules to {len(facts)} facts.")
        
        elif level == ExplanationLevel.SUMMARY:
            explanation_parts.append("The system performed the following analysis:")
            explanation_parts.append(f"1. Loaded {len(facts)} facts about the database schema and query")
            explanation_parts.append(f"2. Applied {len(reasoning_trace)} reasoning rules")
            explanation_parts.append("3. Checked for constraint violations")
            explanation_parts.append("4. Generated confidence score and recommendations")
        
        elif level in [ExplanationLevel.DETAILED, ExplanationLevel.TECHNICAL]:
            explanation_parts.append("Detailed reasoning steps:")
            
            for i, step in enumerate(reasoning_trace, 1):
                explanation_parts.append(f"{i:2d}. {step}")
            
            if level == ExplanationLevel.TECHNICAL:
                explanation_parts.append("")
                explanation_parts.append("Facts analyzed:")
                for i, fact in enumerate(facts[:10], 1):  # Show first 10 facts
                    explanation_parts.append(f"    {fact}")
                
                if len(facts) > 10:
                    explanation_parts.append(f"    ... and {len(facts) - 10} more facts")
        
        return '\n'.join(explanation_parts)
    
    def format_explanation(self, explanation: Explanation,
                          style: ExplanationStyle = ExplanationStyle.NATURAL) -> str:
        """Format explanation for display"""
        if style == ExplanationStyle.NATURAL:
            return self._format_natural(explanation)
        elif style == ExplanationStyle.STRUCTURED:
            return self._format_structured(explanation)
        elif style == ExplanationStyle.TECHNICAL:
            return self._format_technical(explanation)
        elif style == ExplanationStyle.INTERACTIVE:
            return self._format_interactive(explanation)
        else:
            return self._format_natural(explanation)
    
    def _format_natural(self, explanation: Explanation) -> str:
        """Format explanation in natural language style"""
        parts = []
        
        parts.append(explanation.title)
        parts.append("=" * len(explanation.title))
        parts.append("")
        parts.append(explanation.summary)
        parts.append("")
        
        if explanation.details:
            parts.append("Details:")
            for detail in explanation.details:
                parts.append(detail)
            parts.append("")
        
        if explanation.confidence_explanation:
            parts.append(explanation.confidence_explanation)
            parts.append("")
        
        if explanation.recommendations:
            parts.append("Recommendations:")
            for rec in explanation.recommendations:
                parts.append(rec)
            parts.append("")
        
        return '\n'.join(parts)
    
    def _format_structured(self, explanation: Explanation) -> str:
        """Format explanation in structured bullet-point style"""
        parts = []
        
        parts.append(f"# {explanation.title}")
        parts.append("")
        parts.append(f"**Summary:** {explanation.summary}")
        parts.append("")
        
        if explanation.details:
            parts.append("## Details")
            for detail in explanation.details:
                if detail.startswith(('•', '-', '*')):
                    parts.append(detail)
                else:
                    parts.append(f"- {detail}")
            parts.append("")
        
        if explanation.confidence_explanation:
            parts.append(f"## Confidence")
            parts.append(f"- {explanation.confidence_explanation}")
            parts.append("")
        
        if explanation.recommendations:
            parts.append("## Recommendations")
            for rec in explanation.recommendations:
                if rec.startswith(('•', '-', '*')):
                    parts.append(rec)
                else:
                    parts.append(f"- {rec}")
        
        return '\n'.join(parts)
    
    def _format_technical(self, explanation: Explanation) -> str:
        """Format explanation in technical style"""
        parts = []
        
        parts.append(f"VALIDATION_RESULT: {explanation.title}")
        parts.append(f"SUMMARY: {explanation.summary}")
        parts.append("")
        
        if explanation.technical_details:
            parts.append("TECHNICAL_DETAILS:")
            for key, value in explanation.technical_details.items():
                parts.append(f"  {key.upper()}: {value}")
            parts.append("")
        
        if explanation.details:
            parts.append("ANALYSIS_DETAILS:")
            for i, detail in enumerate(explanation.details, 1):
                parts.append(f"  [{i:02d}] {detail}")
            parts.append("")
        
        if explanation.recommendations:
            parts.append("RECOMMENDATIONS:")
            for i, rec in enumerate(explanation.recommendations, 1):
                parts.append(f"  [{i:02d}] {rec}")
        
        return '\n'.join(parts)
    
    def _format_interactive(self, explanation: Explanation) -> str:
        """Format explanation in interactive Q&A style"""
        parts = []
        
        parts.append("❓ What happened with your SQL query?")
        parts.append(f"   {explanation.summary}")
        parts.append("")
        
        if explanation.details and len(explanation.details) > 1:
            parts.append("❓ What specific issues were found?")
            for detail in explanation.details[1:]:  # Skip intro line
                if detail.strip():
                    parts.append(f"   {detail}")
            parts.append("")
        
        if explanation.confidence_explanation:
            parts.append("❓ How confident is this analysis?")
            parts.append(f"   {explanation.confidence_explanation}")
            parts.append("")
        
        if explanation.recommendations:
            parts.append("❓ What should I do next?")
            for rec in explanation.recommendations:
                parts.append(f"   {rec}")
        
        return '\n'.join(parts)
    
    def export_explanation(self, explanation: Explanation, format_type: str = "json") -> str:
        """Export explanation in specified format"""
        if format_type.lower() == "json":
            return json.dumps({
                'title': explanation.title,
                'summary': explanation.summary,
                'details': explanation.details,
                'recommendations': explanation.recommendations,
                'confidence_explanation': explanation.confidence_explanation,
                'technical_details': explanation.technical_details,
                'references': explanation.references,
                'timestamp': explanation.timestamp
            }, indent=2)
        else:
            return self.format_explanation(explanation)


# Convenience functions
def create_explanation_generator() -> ExplanationGenerator:
    """Create a new explanation generator"""
    return ExplanationGenerator()


def generate_simple_explanation(query: str, violations: List[Violation],
                               confidence: float = 0.8) -> str:
    """Generate a simple explanation for query validation"""
    generator = ExplanationGenerator()
    context = ExplanationContext(
        query=query,
        violations=violations,
        confidence=confidence
    )
    explanation = generator.generate_explanation(context)
    return generator.format_explanation(explanation)