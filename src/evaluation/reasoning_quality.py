#!/usr/bin/env python3
"""
Reasoning Quality Assessment Module

This module provides comprehensive evaluation of reasoning quality for 
neurosymbolic SQL generation, including explanation coherence, completeness,
and faithfulness evaluation.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import difflib
import math

@dataclass
class ReasoningEvaluationResult:
    """Result container for reasoning quality evaluation"""
    
    explanation_coherence: float
    explanation_completeness: float
    explanation_accuracy: float
    reasoning_depth: float
    technical_correctness: float
    clarity_score: float
    overall_score: float
    
    # Detailed analysis
    coherence_issues: List[str]
    completeness_gaps: List[str]
    accuracy_problems: List[str]
    technical_errors: List[str]
    clarity_problems: List[str]
    
    # Metrics
    word_count: int
    sentence_count: int
    technical_terms_count: int
    reasoning_steps: int

class ReasoningQualityAssessment:
    """
    Comprehensive reasoning quality evaluation
    
    Evaluates multiple aspects of reasoning quality:
    1. Explanation coherence and logical flow
    2. Completeness of reasoning coverage
    3. Accuracy and faithfulness to the query
    4. Technical correctness of explanations
    5. Clarity and understandability
    6. Reasoning depth and sophistication
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reasoning quality assessment
        
        Args:
            config: Configuration including:
                - explanation_methods: List of evaluation methods to use
                - enable_human_evaluation: Whether human evaluation is available
                - strict_mode: Whether to use strict evaluation criteria
                - technical_vocabulary: Custom technical terms to recognize
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Configuration
        self.explanation_methods = config.get('explanation_methods', ['coherence', 'completeness', 'accuracy'])
        self.enable_human_eval = config.get('enable_human_evaluation', False)
        self.strict_mode = config.get('strict_mode', False)
        
        # Technical vocabulary for SQL domain
        self.sql_technical_terms = self._load_sql_technical_terms()
        self.sql_technical_terms.update(config.get('technical_vocabulary', {}))
        
        # Reasoning patterns
        self.reasoning_patterns = self._load_reasoning_patterns()
        
        self.logger.info("Reasoning quality assessment initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for reasoning quality assessment"""
        logger = logging.getLogger("reasoning_quality")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_sql_technical_terms(self) -> Dict[str, str]:
        """Load SQL technical vocabulary"""
        
        return {
            # SQL Keywords
            'select': 'SQL query operation',
            'from': 'table specification clause',
            'where': 'filtering condition clause',
            'join': 'table relationship operation',
            'inner join': 'matching records join',
            'left join': 'all left table records join',
            'right join': 'all right table records join',
            'group by': 'grouping aggregation clause',
            'having': 'group filtering clause',
            'order by': 'sorting clause',
            'limit': 'result quantity restriction',
            
            # SQL Functions
            'count': 'counting aggregation function',
            'sum': 'summation aggregation function',
            'avg': 'average aggregation function',
            'max': 'maximum value function',
            'min': 'minimum value function',
            
            # Database Concepts
            'table': 'database table structure',
            'column': 'table attribute field',
            'row': 'table record entry',
            'primary key': 'unique identifier constraint',
            'foreign key': 'referential integrity constraint',
            'index': 'query optimization structure',
            'constraint': 'data integrity rule',
            'schema': 'database structure definition',
            
            # Query Concepts
            'subquery': 'nested query expression',
            'aggregation': 'data summarization operation',
            'filtering': 'data selection criteria',
            'sorting': 'result ordering operation',
            'grouping': 'data categorization operation'
        }
    
    def _load_reasoning_patterns(self) -> List[Dict[str, Any]]:
        """Load common reasoning patterns for SQL explanations"""
        
        return [
            {
                'name': 'step_by_step',
                'pattern': r'(first|then|next|finally|step \d+)',
                'description': 'Sequential step-by-step reasoning'
            },
            {
                'name': 'conditional_reasoning',
                'pattern': r'(if|when|because|since|due to)',
                'description': 'Conditional or causal reasoning'
            },
            {
                'name': 'comparison',
                'pattern': r'(compare|versus|different|similar|unlike)',
                'description': 'Comparative reasoning'
            },
            {
                'name': 'explanation',
                'pattern': r'(this means|in other words|specifically|namely)',
                'description': 'Explanatory reasoning'
            },
            {
                'name': 'justification',
                'pattern': r'(because|since|as|given that|considering)',
                'description': 'Justification reasoning'
            }
        ]
    
    def evaluate_reasoning(
        self,
        explanation: str,
        generated_sql: str,
        instruction: str,
        schema: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive reasoning quality evaluation
        
        Args:
            explanation: Generated explanation text
            generated_sql: Generated SQL query
            instruction: Original instruction/question
            schema: Optional schema information
            
        Returns:
            Dictionary containing all reasoning quality metrics
        """
        
        if not explanation or not explanation.strip():
            return self._create_empty_result("No explanation provided")
        
        # Initialize analysis containers
        coherence_issues = []
        completeness_gaps = []
        accuracy_problems = []
        technical_errors = []
        clarity_problems = []
        
        # 1. Explanation coherence
        coherence_score, coherence_issues = self._evaluate_coherence(explanation)
        
        # 2. Explanation completeness
        completeness_score, completeness_gaps = self._evaluate_completeness(
            explanation, generated_sql, instruction, schema
        )
        
        # 3. Explanation accuracy
        accuracy_score, accuracy_problems = self._evaluate_accuracy(
            explanation, generated_sql, instruction
        )
        
        # 4. Technical correctness
        technical_score, technical_errors = self._evaluate_technical_correctness(
            explanation, generated_sql
        )
        
        # 5. Clarity and understandability
        clarity_score, clarity_problems = self._evaluate_clarity(explanation)
        
        # 6. Reasoning depth
        depth_score, reasoning_steps = self._evaluate_reasoning_depth(explanation)
        
        # Calculate metrics
        metrics = self._calculate_explanation_metrics(explanation)
        
        # Calculate overall score
        overall_score = self._calculate_overall_reasoning_score({
            'coherence': coherence_score,
            'completeness': completeness_score,
            'accuracy': accuracy_score,
            'technical_correctness': technical_score,
            'clarity': clarity_score,
            'depth': depth_score
        })
        
        # Create comprehensive result
        result = ReasoningEvaluationResult(
            explanation_coherence=coherence_score,
            explanation_completeness=completeness_score,
            explanation_accuracy=accuracy_score,
            reasoning_depth=depth_score,
            technical_correctness=technical_score,
            clarity_score=clarity_score,
            overall_score=overall_score,
            
            # Detailed analysis
            coherence_issues=coherence_issues,
            completeness_gaps=completeness_gaps,
            accuracy_problems=accuracy_problems,
            technical_errors=technical_errors,
            clarity_problems=clarity_problems,
            
            # Metrics
            word_count=metrics['word_count'],
            sentence_count=metrics['sentence_count'],
            technical_terms_count=metrics['technical_terms_count'],
            reasoning_steps=reasoning_steps
        )
        
        return self._result_to_dict(result)
    
    def _evaluate_coherence(self, explanation: str) -> Tuple[float, List[str]]:
        """Evaluate explanation coherence and logical flow"""
        
        issues = []
        coherence_score = 1.0
        
        # Split into sentences
        sentences = self._split_into_sentences(explanation)
        
        if len(sentences) < 2:
            issues.append("Explanation too short for coherence evaluation")
            return 0.5, issues
        
        # Check for logical connectors
        logical_connectors = ['because', 'since', 'therefore', 'thus', 'however', 'moreover', 'furthermore', 'consequently']
        connector_count = sum(1 for sentence in sentences for connector in logical_connectors if connector in sentence.lower())
        
        if connector_count == 0 and len(sentences) > 2:
            issues.append("Lacks logical connectors between ideas")
            coherence_score -= 0.2
        
        # Check for consistent pronoun usage
        pronoun_consistency = self._check_pronoun_consistency(sentences)
        if not pronoun_consistency:
            issues.append("Inconsistent pronoun usage")
            coherence_score -= 0.1
        
        # Check for topic continuity
        topic_continuity = self._check_topic_continuity(sentences)
        if topic_continuity < 0.7:
            issues.append("Poor topic continuity between sentences")
            coherence_score -= 0.2
        
        # Check for contradiction
        contradictions = self._detect_contradictions(sentences)
        if contradictions:
            issues.extend(contradictions)
            coherence_score -= 0.3 * len(contradictions)
        
        return max(0.0, coherence_score), issues
    
    def _evaluate_completeness(
        self,
        explanation: str,
        generated_sql: str,
        instruction: str,
        schema: Optional[str] = None
    ) -> Tuple[float, List[str]]:
        """Evaluate explanation completeness"""
        
        gaps = []
        completeness_score = 1.0
        
        # Extract SQL components for coverage check
        sql_components = self._extract_sql_components(generated_sql)
        
        # Check coverage of major SQL components
        coverage_checks = {
            'SELECT clause': self._explanation_covers_select(explanation, sql_components),
            'FROM clause': self._explanation_covers_from(explanation, sql_components),
            'WHERE clause': self._explanation_covers_where(explanation, sql_components),
            'JOIN operations': self._explanation_covers_joins(explanation, sql_components),
            'GROUP BY clause': self._explanation_covers_groupby(explanation, sql_components),
            'ORDER BY clause': self._explanation_covers_orderby(explanation, sql_components)
        }
        
        missing_coverage = []
        for component, covered in coverage_checks.items():
            if not covered and component.split()[0].upper() in generated_sql.upper():
                missing_coverage.append(component)
        
        if missing_coverage:
            gaps.extend([f"Missing explanation for {comp}" for comp in missing_coverage])
            completeness_score -= 0.15 * len(missing_coverage)
        
        # Check if explanation addresses the original instruction
        instruction_coverage = self._check_instruction_coverage(explanation, instruction)
        if instruction_coverage < 0.5:
            gaps.append("Explanation doesn't adequately address the original instruction")
            completeness_score -= 0.3
        
        # Check for reasoning process coverage
        reasoning_coverage = self._check_reasoning_process_coverage(explanation)
        if reasoning_coverage < 0.6:
            gaps.append("Lacks detailed reasoning process explanation")
            completeness_score -= 0.2
        
        return max(0.0, completeness_score), gaps
    
    def _evaluate_accuracy(
        self,
        explanation: str,
        generated_sql: str,
        instruction: str
    ) -> Tuple[float, List[str]]:
        """Evaluate explanation accuracy and faithfulness"""
        
        problems = []
        accuracy_score = 1.0
        
        # Check for factual accuracy about SQL components
        sql_facts = self._extract_sql_facts_from_explanation(explanation)
        sql_reality = self._extract_actual_sql_facts(generated_sql)
        
        # Compare explained facts with actual SQL
        fact_mismatches = self._compare_sql_facts(sql_facts, sql_reality)
        if fact_mismatches:
            problems.extend(fact_mismatches)
            accuracy_score -= 0.2 * len(fact_mismatches)
        
        # Check for misleading statements
        misleading_statements = self._detect_misleading_statements(explanation, generated_sql)
        if misleading_statements:
            problems.extend(misleading_statements)
            accuracy_score -= 0.3 * len(misleading_statements)
        
        # Check instruction-explanation alignment
        alignment_score = self._check_instruction_explanation_alignment(explanation, instruction)
        if alignment_score < 0.7:
            problems.append("Explanation doesn't align well with the original instruction")
            accuracy_score -= 0.2
        
        return max(0.0, accuracy_score), problems
    
    def _evaluate_technical_correctness(
        self,
        explanation: str,
        generated_sql: str
    ) -> Tuple[float, List[str]]:
        """Evaluate technical correctness of explanation"""
        
        errors = []
        technical_score = 1.0
        
        # Check for correct SQL terminology usage
        terminology_errors = self._check_sql_terminology(explanation)
        if terminology_errors:
            errors.extend(terminology_errors)
            technical_score -= 0.1 * len(terminology_errors)
        
        # Check for technical accuracy of SQL descriptions
        technical_descriptions = self._extract_technical_descriptions(explanation)
        description_errors = self._validate_technical_descriptions(technical_descriptions, generated_sql)
        if description_errors:
            errors.extend(description_errors)
            technical_score -= 0.2 * len(description_errors)
        
        # Check for SQL best practices mentions
        best_practices_score = self._evaluate_best_practices_coverage(explanation, generated_sql)
        if best_practices_score < 0.3:
            errors.append("Limited coverage of SQL best practices or considerations")
            technical_score -= 0.1
        
        return max(0.0, technical_score), errors
    
    def _evaluate_clarity(self, explanation: str) -> Tuple[float, List[str]]:
        """Evaluate clarity and understandability of explanation"""
        
        problems = []
        clarity_score = 1.0
        
        # Check sentence length and complexity
        sentences = self._split_into_sentences(explanation)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        if avg_sentence_length > 25:
            problems.append("Sentences are too long and complex")
            clarity_score -= 0.2
        
        # Check for jargon without explanation
        unexplained_jargon = self._find_unexplained_jargon(explanation)
        if unexplained_jargon:
            problems.extend([f"Unexplained technical term: {term}" for term in unexplained_jargon])
            clarity_score -= 0.1 * len(unexplained_jargon)
        
        # Check for readability
        readability_score = self._calculate_readability(explanation)
        if readability_score < 0.6:
            problems.append("Low readability score - explanation may be hard to understand")
            clarity_score -= 0.2
        
        # Check for structure and organization
        structure_score = self._evaluate_explanation_structure(explanation)
        if structure_score < 0.5:
            problems.append("Poor explanation structure and organization")
            clarity_score -= 0.15
        
        return max(0.0, clarity_score), problems
    
    def _evaluate_reasoning_depth(self, explanation: str) -> Tuple[float, int]:
        """Evaluate depth and sophistication of reasoning"""
        
        # Count reasoning steps
        reasoning_steps = 0
        
        # Look for reasoning patterns
        for pattern_info in self.reasoning_patterns:
            matches = re.findall(pattern_info['pattern'], explanation, re.IGNORECASE)
            reasoning_steps += len(matches)
        
        # Look for explicit step indicators
        step_indicators = re.findall(r'(step \d+|first|second|third|then|next|finally)', explanation, re.IGNORECASE)
        reasoning_steps += len(step_indicators)
        
        # Look for causal relationships
        causal_indicators = re.findall(r'(because|since|therefore|thus|as a result|consequently)', explanation, re.IGNORECASE)
        reasoning_steps += len(causal_indicators)
        
        # Calculate depth score based on reasoning complexity
        depth_score = min(1.0, reasoning_steps / 5.0)  # Normalize to 0-1
        
        # Bonus for sophisticated reasoning
        if 'optimization' in explanation.lower() or 'efficiency' in explanation.lower():
            depth_score = min(1.0, depth_score + 0.1)
        
        if 'constraint' in explanation.lower() or 'requirement' in explanation.lower():
            depth_score = min(1.0, depth_score + 0.1)
        
        return depth_score, reasoning_steps
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _check_pronoun_consistency(self, sentences: List[str]) -> bool:
        """Check for consistent pronoun usage"""
        
        # Simple check for consistent use of "we", "this", "the query", etc.
        pronoun_usage = {'we': 0, 'this': 0, 'the query': 0, 'it': 0}
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for pronoun in pronoun_usage:
                if pronoun in sentence_lower:
                    pronoun_usage[pronoun] += 1
        
        # Check if there's a dominant pronoun style
        total_pronouns = sum(pronoun_usage.values())
        if total_pronouns > 0:
            max_usage = max(pronoun_usage.values())
            consistency = max_usage / total_pronouns
            return consistency > 0.6
        
        return True  # No pronouns to check
    
    def _check_topic_continuity(self, sentences: List[str]) -> float:
        """Check topic continuity between sentences"""
        
        if len(sentences) < 2:
            return 1.0
        
        continuity_scores = []
        
        for i in range(len(sentences) - 1):
            current_words = set(sentences[i].lower().split())
            next_words = set(sentences[i + 1].lower().split())
            
            # Calculate word overlap
            overlap = len(current_words.intersection(next_words))
            total_unique = len(current_words.union(next_words))
            
            if total_unique > 0:
                continuity = overlap / total_unique
                continuity_scores.append(continuity)
        
        return sum(continuity_scores) / len(continuity_scores) if continuity_scores else 0.0
    
    def _detect_contradictions(self, sentences: List[str]) -> List[str]:
        """Detect contradictions in explanation"""
        
        contradictions = []
        
        # Simple contradiction detection
        positive_statements = []
        negative_statements = []
        
        for sentence in sentences:
            if any(neg in sentence.lower() for neg in ['not', 'no', 'never', 'without']):
                negative_statements.append(sentence)
            else:
                positive_statements.append(sentence)
        
        # This is a simplified implementation
        # In practice, would need more sophisticated contradiction detection
        
        return contradictions
    
    def _extract_sql_components(self, sql: str) -> Dict[str, Any]:
        """Extract SQL components for coverage analysis"""
        
        sql_upper = sql.upper()
        
        return {
            'has_select': 'SELECT' in sql_upper,
            'has_from': 'FROM' in sql_upper,
            'has_where': 'WHERE' in sql_upper,
            'has_join': any(join in sql_upper for join in ['JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN']),
            'has_group_by': 'GROUP BY' in sql_upper,
            'has_order_by': 'ORDER BY' in sql_upper,
            'has_having': 'HAVING' in sql_upper,
            'has_limit': 'LIMIT' in sql_upper,
            'has_aggregation': any(agg in sql_upper for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN'])
        }
    
    def _explanation_covers_select(self, explanation: str, components: Dict[str, Any]) -> bool:
        """Check if explanation covers SELECT clause"""
        
        if not components['has_select']:
            return True
        
        select_indicators = ['select', 'column', 'field', 'attribute', 'retrieve', 'return']
        explanation_lower = explanation.lower()
        
        return any(indicator in explanation_lower for indicator in select_indicators)
    
    def _explanation_covers_from(self, explanation: str, components: Dict[str, Any]) -> bool:
        """Check if explanation covers FROM clause"""
        
        if not components['has_from']:
            return True
        
        from_indicators = ['from', 'table', 'source', 'data']
        explanation_lower = explanation.lower()
        
        return any(indicator in explanation_lower for indicator in from_indicators)
    
    def _explanation_covers_where(self, explanation: str, components: Dict[str, Any]) -> bool:
        """Check if explanation covers WHERE clause"""
        
        if not components['has_where']:
            return True
        
        where_indicators = ['where', 'filter', 'condition', 'criteria', 'restrict']
        explanation_lower = explanation.lower()
        
        return any(indicator in explanation_lower for indicator in where_indicators)
    
    def _explanation_covers_joins(self, explanation: str, components: Dict[str, Any]) -> bool:
        """Check if explanation covers JOIN operations"""
        
        if not components['has_join']:
            return True
        
        join_indicators = ['join', 'combine', 'merge', 'connect', 'relationship']
        explanation_lower = explanation.lower()
        
        return any(indicator in explanation_lower for indicator in join_indicators)
    
    def _explanation_covers_groupby(self, explanation: str, components: Dict[str, Any]) -> bool:
        """Check if explanation covers GROUP BY clause"""
        
        if not components['has_group_by']:
            return True
        
        group_indicators = ['group', 'aggregate', 'summarize', 'category']
        explanation_lower = explanation.lower()
        
        return any(indicator in explanation_lower for indicator in group_indicators)
    
    def _explanation_covers_orderby(self, explanation: str, components: Dict[str, Any]) -> bool:
        """Check if explanation covers ORDER BY clause"""
        
        if not components['has_order_by']:
            return True
        
        order_indicators = ['order', 'sort', 'arrange', 'sequence']
        explanation_lower = explanation.lower()
        
        return any(indicator in explanation_lower for indicator in order_indicators)
    
    def _check_instruction_coverage(self, explanation: str, instruction: str) -> float:
        """Check how well explanation covers the original instruction"""
        
        # Extract key terms from instruction
        instruction_words = set(instruction.lower().split())
        explanation_words = set(explanation.lower().split())
        
        # Calculate coverage
        covered_words = instruction_words.intersection(explanation_words)
        coverage = len(covered_words) / len(instruction_words) if instruction_words else 0.0
        
        return coverage
    
    def _check_reasoning_process_coverage(self, explanation: str) -> float:
        """Check coverage of reasoning process"""
        
        reasoning_indicators = [
            'analyze', 'determine', 'identify', 'consider', 'evaluate',
            'process', 'approach', 'method', 'strategy', 'logic'
        ]
        
        explanation_lower = explanation.lower()
        covered_indicators = sum(1 for indicator in reasoning_indicators if indicator in explanation_lower)
        
        return covered_indicators / len(reasoning_indicators)
    
    def _extract_sql_facts_from_explanation(self, explanation: str) -> List[str]:
        """Extract factual claims about SQL from explanation"""
        
        # This is a simplified implementation
        # Would extract specific claims about what the SQL does
        
        facts = []
        sentences = self._split_into_sentences(explanation)
        
        for sentence in sentences:
            if any(verb in sentence.lower() for verb in ['select', 'filter', 'join', 'group', 'order']):
                facts.append(sentence.strip())
        
        return facts
    
    def _extract_actual_sql_facts(self, sql: str) -> List[str]:
        """Extract factual information from actual SQL"""
        
        # This would analyze the SQL and extract what it actually does
        facts = []
        sql_upper = sql.upper()
        
        if 'SELECT' in sql_upper:
            facts.append("Query performs data selection")
        
        if 'WHERE' in sql_upper:
            facts.append("Query applies filtering conditions")
        
        if 'JOIN' in sql_upper:
            facts.append("Query combines multiple tables")
        
        if 'GROUP BY' in sql_upper:
            facts.append("Query performs data grouping")
        
        return facts
    
    def _compare_sql_facts(self, explained_facts: List[str], actual_facts: List[str]) -> List[str]:
        """Compare explained facts with actual SQL facts"""
        
        # Simplified fact comparison
        mismatches = []
        
        # This would be more sophisticated in practice
        # For now, just check basic consistency
        
        return mismatches
    
    def _detect_misleading_statements(self, explanation: str, sql: str) -> List[str]:
        """Detect misleading statements in explanation"""
        
        misleading = []
        
        # Check for common misleading patterns
        explanation_lower = explanation.lower()
        sql_upper = sql.upper()
        
        if 'all records' in explanation_lower and 'WHERE' in sql_upper:
            misleading.append("Claims to select all records but has WHERE clause")
        
        if 'no filtering' in explanation_lower and 'WHERE' in sql_upper:
            misleading.append("Claims no filtering but has WHERE clause")
        
        return misleading
    
    def _check_instruction_explanation_alignment(self, explanation: str, instruction: str) -> float:
        """Check alignment between instruction and explanation"""
        
        # Use semantic similarity (simplified)
        instruction_words = set(instruction.lower().split())
        explanation_words = set(explanation.lower().split())
        
        if not instruction_words:
            return 1.0
        
        overlap = len(instruction_words.intersection(explanation_words))
        alignment = overlap / len(instruction_words)
        
        return alignment
    
    def _check_sql_terminology(self, explanation: str) -> List[str]:
        """Check for correct SQL terminology usage"""
        
        errors = []
        
        # Check for common terminology mistakes
        explanation_lower = explanation.lower()
        
        # This would be expanded with more comprehensive checks
        if 'field' in explanation_lower and 'column' not in explanation_lower:
            # Prefer 'column' over 'field' in SQL context
            pass  # Not necessarily an error
        
        return errors
    
    def _extract_technical_descriptions(self, explanation: str) -> List[str]:
        """Extract technical descriptions from explanation"""
        
        descriptions = []
        sentences = self._split_into_sentences(explanation)
        
        for sentence in sentences:
            if any(term in sentence.lower() for term in self.sql_technical_terms.keys()):
                descriptions.append(sentence)
        
        return descriptions
    
    def _validate_technical_descriptions(self, descriptions: List[str], sql: str) -> List[str]:
        """Validate technical descriptions against actual SQL"""
        
        errors = []
        
        # This would validate that technical descriptions are accurate
        # Simplified implementation for now
        
        return errors
    
    def _evaluate_best_practices_coverage(self, explanation: str, sql: str) -> float:
        """Evaluate coverage of SQL best practices"""
        
        best_practices_indicators = [
            'performance', 'efficiency', 'optimization', 'index', 'constraint',
            'security', 'readability', 'maintainability'
        ]
        
        explanation_lower = explanation.lower()
        covered_practices = sum(1 for practice in best_practices_indicators if practice in explanation_lower)
        
        return covered_practices / len(best_practices_indicators)
    
    def _find_unexplained_jargon(self, explanation: str) -> List[str]:
        """Find technical terms that aren't explained"""
        
        unexplained = []
        
        for term in self.sql_technical_terms.keys():
            if term in explanation.lower():
                # Check if there's an explanation nearby
                term_context = self._get_term_context(explanation, term)
                if not self._has_explanation_context(term_context, term):
                    unexplained.append(term)
        
        return unexplained
    
    def _get_term_context(self, explanation: str, term: str) -> str:
        """Get context around a technical term"""
        
        # Get sentence containing the term
        sentences = self._split_into_sentences(explanation)
        
        for sentence in sentences:
            if term in sentence.lower():
                return sentence
        
        return ""
    
    def _has_explanation_context(self, context: str, term: str) -> bool:
        """Check if context provides explanation for term"""
        
        explanation_indicators = ['means', 'is', 'refers to', 'which', 'that']
        context_lower = context.lower()
        
        return any(indicator in context_lower for indicator in explanation_indicators)
    
    def _calculate_readability(self, explanation: str) -> float:
        """Calculate readability score (simplified Flesch-like metric)"""
        
        sentences = self._split_into_sentences(explanation)
        words = explanation.split()
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Simple readability calculation
        # Lower sentence length = higher readability
        readability = max(0.0, 1.0 - (avg_sentence_length - 10) / 20)
        
        return min(1.0, readability)
    
    def _evaluate_explanation_structure(self, explanation: str) -> float:
        """Evaluate structure and organization of explanation"""
        
        sentences = self._split_into_sentences(explanation)
        
        if len(sentences) < 2:
            return 0.5
        
        structure_score = 0.0
        
        # Check for introduction
        first_sentence = sentences[0].lower()
        if any(intro in first_sentence for intro in ['this query', 'we', 'the sql', 'to']):
            structure_score += 0.3
        
        # Check for logical flow indicators
        flow_indicators = ['first', 'then', 'next', 'finally', 'therefore']
        has_flow = any(any(indicator in sentence.lower() for indicator in flow_indicators) for sentence in sentences)
        if has_flow:
            structure_score += 0.4
        
        # Check for conclusion or summary
        last_sentence = sentences[-1].lower()
        if any(conclusion in last_sentence for conclusion in ['result', 'output', 'final', 'therefore']):
            structure_score += 0.3
        
        return structure_score
    
    def _calculate_explanation_metrics(self, explanation: str) -> Dict[str, int]:
        """Calculate basic metrics about explanation"""
        
        words = explanation.split()
        sentences = self._split_into_sentences(explanation)
        
        # Count technical terms
        technical_count = 0
        explanation_lower = explanation.lower()
        for term in self.sql_technical_terms.keys():
            technical_count += explanation_lower.count(term)
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'technical_terms_count': technical_count
        }
    
    def _calculate_overall_reasoning_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall reasoning score"""
        
        weights = {
            'coherence': 0.20,
            'completeness': 0.25,
            'accuracy': 0.25,
            'technical_correctness': 0.15,
            'clarity': 0.10,
            'depth': 0.05
        }
        
        overall_score = 0.0
        total_weight = 0.0
        
        for aspect, weight in weights.items():
            if aspect in scores:
                overall_score += scores[aspect] * weight
                total_weight += weight
        
        return overall_score / total_weight if total_weight > 0 else 0.0
    
    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """Create empty result for cases with no explanation"""
        
        return {
            'explanation_coherence': 0.0,
            'explanation_completeness': 0.0,
            'explanation_accuracy': 0.0,
            'reasoning_depth': 0.0,
            'technical_correctness': 0.0,
            'clarity_score': 0.0,
            'overall_score': 0.0,
            
            'coherence_issues': [reason],
            'completeness_gaps': [reason],
            'accuracy_problems': [reason],
            'technical_errors': [reason],
            'clarity_problems': [reason],
            
            'word_count': 0,
            'sentence_count': 0,
            'technical_terms_count': 0,
            'reasoning_steps': 0
        }
    
    def _result_to_dict(self, result: ReasoningEvaluationResult) -> Dict[str, Any]:
        """Convert result to dictionary"""
        
        return {
            'explanation_coherence': result.explanation_coherence,
            'explanation_completeness': result.explanation_completeness,
            'explanation_accuracy': result.explanation_accuracy,
            'reasoning_depth': result.reasoning_depth,
            'technical_correctness': result.technical_correctness,
            'clarity_score': result.clarity_score,
            'overall_score': result.overall_score,
            
            'detailed_analysis': {
                'coherence_issues': result.coherence_issues,
                'completeness_gaps': result.completeness_gaps,
                'accuracy_problems': result.accuracy_problems,
                'technical_errors': result.technical_errors,
                'clarity_problems': result.clarity_problems
            },
            
            'metrics': {
                'word_count': result.word_count,
                'sentence_count': result.sentence_count,
                'technical_terms_count': result.technical_terms_count,
                'reasoning_steps': result.reasoning_steps
            }
        }
    
    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate reasoning quality results from multiple evaluations"""
        
        if not results:
            return {}
        
        # Calculate averages
        metrics = [
            'explanation_coherence', 'explanation_completeness', 'explanation_accuracy',
            'reasoning_depth', 'technical_correctness', 'clarity_score', 'overall_score'
        ]
        
        aggregated = {}
        
        for metric in metrics:
            values = [r.get(metric, 0.0) for r in results if metric in r]
            if values:
                aggregated[metric] = sum(values) / len(values)
            else:
                aggregated[metric] = 0.0
        
        # Additional aggregate metrics
        aggregated['total_evaluations'] = len(results)
        aggregated['high_quality_explanations'] = sum(1 for r in results if r.get('overall_score', 0.0) > 0.7)
        aggregated['high_quality_rate'] = aggregated['high_quality_explanations'] / len(results) if results else 0.0
        
        # Average metrics
        word_counts = [r.get('metrics', {}).get('word_count', 0) for r in results]
        sentence_counts = [r.get('metrics', {}).get('sentence_count', 0) for r in results]
        technical_counts = [r.get('metrics', {}).get('technical_terms_count', 0) for r in results]
        
        aggregated['avg_word_count'] = sum(word_counts) / len(word_counts) if word_counts else 0.0
        aggregated['avg_sentence_count'] = sum(sentence_counts) / len(sentence_counts) if sentence_counts else 0.0
        aggregated['avg_technical_terms'] = sum(technical_counts) / len(technical_counts) if technical_counts else 0.0
        
        return aggregated