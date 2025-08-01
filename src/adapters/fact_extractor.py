#!/usr/bin/env python3
"""
Fact Extractor

Advanced symbolic fact generation from neural representations.
Converts neural embeddings to structured logical facts for reasoning.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import re
import json
from enum import Enum


class FactType(Enum):
    """Types of extractable facts"""
    SCHEMA_FACT = "schema"
    QUERY_FACT = "query"
    CONSTRAINT_FACT = "constraint"
    SEMANTIC_FACT = "semantic"
    OPERATIONAL_FACT = "operational"


@dataclass
class ExtractedFact:
    """Container for extracted symbolic fact"""
    fact_string: str  # Logical fact representation
    fact_type: FactType  # Type of fact
    confidence: float  # Extraction confidence [0, 1]
    source_tokens: Optional[List[int]] = None  # Source token positions
    attributes: Optional[Dict[str, Any]] = None  # Additional attributes
    reasoning_path: Optional[List[str]] = None  # How this fact was derived


@dataclass
class FactExtractionResult:
    """Result from fact extraction"""
    facts: List[ExtractedFact]  # Extracted facts
    fact_counts: Dict[FactType, int]  # Count by type
    extraction_confidence: float  # Overall extraction confidence
    processing_time: Optional[float] = None  # Processing time in seconds
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata


class SQLPatternMatcher:
    """Pattern matching for SQL-specific fact extraction"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # SQL keyword patterns
        self.sql_patterns = {
            'select_pattern': r'\b(select)\b.*?\bfrom\b',
            'from_pattern': r'\bfrom\b\s+(\w+)',
            'where_pattern': r'\bwhere\b\s+(.+?)(?:\bgroup\b|\border\b|\blimit\b|$)',
            'join_pattern': r'\b(inner|left|right|full)?\s*join\b\s+(\w+)',
            'aggregate_pattern': r'\b(count|sum|avg|min|max|stddev)\s*\(',
            'column_pattern': r'\b(\w+)\.(\w+)\b',
            'comparison_pattern': r'(\w+)\s*(=|!=|<|>|<=|>=|like|in)\s*',
            'logical_pattern': r'\b(and|or|not)\b'
        }
        
        # Compile patterns
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.sql_patterns.items()
        }
    
    def extract_sql_facts(self, text: str) -> List[ExtractedFact]:
        """Extract SQL-specific facts from text"""
        facts = []
        text_lower = text.lower()
        
        # Extract SELECT facts
        if 'select' in text_lower:
            facts.append(ExtractedFact(
                fact_string="query_type(select)",
                fact_type=FactType.QUERY_FACT,
                confidence=0.9
            ))
        
        # Extract table references
        for match in self.compiled_patterns['from_pattern'].finditer(text):
            table_name = match.group(1)
            facts.append(ExtractedFact(
                fact_string=f"query_references_table({table_name})",
                fact_type=FactType.QUERY_FACT,
                confidence=0.85
            ))
        
        # Extract JOIN facts
        for match in self.compiled_patterns['join_pattern'].finditer(text):
            join_type = match.group(1) or 'inner'
            table_name = match.group(2)
            facts.append(ExtractedFact(
                fact_string=f"query_has_join({table_name}, {join_type})",
                fact_type=FactType.QUERY_FACT,
                confidence=0.8
            ))
        
        # Extract aggregate functions
        for match in self.compiled_patterns['aggregate_pattern'].finditer(text):
            agg_function = match.group(1)
            facts.append(ExtractedFact(
                fact_string=f"query_uses_aggregate({agg_function})",
                fact_type=FactType.OPERATIONAL_FACT,
                confidence=0.85
            ))
        
        # Extract column references
        for match in self.compiled_patterns['column_pattern'].finditer(text):
            table, column = match.groups()
            facts.append(ExtractedFact(
                fact_string=f"query_references_column({table}, {column})",
                fact_type=FactType.QUERY_FACT,
                confidence=0.8
            ))
        
        return facts


class SemanticFactExtractor(nn.Module):
    """Neural network for semantic fact extraction"""
    
    def __init__(self, hidden_dim: int = 4096, fact_vocab_size: int = 200):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fact_vocab_size = fact_vocab_size
        
        # Fact classification heads
        self.fact_classifiers = nn.ModuleDict({
            'schema': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 50),  # 50 schema fact types
                nn.Sigmoid()
            ),
            'query': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 80),  # 80 query fact types
                nn.Sigmoid()
            ),
            'constraint': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 30),  # 30 constraint fact types
                nn.Sigmoid()
            ),
            'semantic': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 40),  # 40 semantic fact types
                nn.Sigmoid()
            )
        })
        
        # Fact template vocabulary
        self.fact_templates = self._initialize_fact_templates()
        
    def _initialize_fact_templates(self) -> Dict[str, List[str]]:
        """Initialize fact templates for different categories"""
        return {
            'schema': [
                'table_exists({})', 'column_exists({}, {})', 'column_type({}, {}, {})',
                'primary_key({}, {})', 'foreign_key({}, {}, {}, {})', 'unique_constraint({}, {})',
                'not_null_constraint({}, {})', 'index_exists({}, {})', 'view_exists({})',
                'trigger_exists({}, {})', 'procedure_exists({})', 'function_exists({})',
                'schema_contains_table({}, {})', 'table_has_column({}, {})', 'column_has_type({}, {}, {})',
                'table_has_primary_key({}, {})', 'table_has_foreign_key({}, {}, {}, {})',
                'column_is_nullable({}, {})', 'column_is_unique({}, {})', 'table_has_index({}, {})',
                'database_contains_schema({}, {})', 'schema_version({}, {})', 'table_created_date({}, {})',
                'column_default_value({}, {}, {})', 'constraint_name({}, {})', 'index_type({}, {}, {})',
                'foreign_key_action({}, {}, {}, {}, {})', 'check_constraint({}, {}, {})',
                'table_comment({}, {})', 'column_comment({}, {}, {})', 'table_engine({}, {})',
                'table_charset({}, {})', 'column_collation({}, {}, {})', 'partition_exists({}, {})',
                'table_row_count({}, {})', 'table_size_bytes({}, {})', 'column_distinct_count({}, {}, {})',
                'column_null_count({}, {}, {})', 'table_last_updated({}, {})', 'column_min_value({}, {}, {})',
                'column_max_value({}, {}, {})', 'table_has_trigger({}, {})', 'view_definition({}, {})',
                'materialized_view({})' 'temporary_table({})', 'external_table({})', 'partitioned_table({})',
                'clustered_table({})', 'compressed_table({})', 'encrypted_table({})', 'audit_table({})'
            ],
            'query': [
                'query_type({})', 'query_references_table({})', 'query_references_column({}, {})',
                'query_has_join({}, {})', 'query_has_condition({}, {})', 'query_uses_aggregate({})',
                'query_has_group_by({})', 'query_has_order_by({})', 'query_has_limit({})',
                'query_has_subquery({})', 'query_has_union({})', 'query_has_window_function({})',
                'select_column({}, {})', 'from_table({})', 'where_condition({}, {})',
                'join_condition({}, {}, {})', 'group_by_column({}, {})', 'order_by_column({}, {}, {})',
                'having_condition({}, {})', 'limit_value({})', 'offset_value({})',
                'insert_into_table({})', 'insert_column({}, {})', 'insert_value({}, {}, {})',
                'update_table({})', 'update_column({}, {})', 'update_value({}, {}, {})',
                'delete_from_table({})', 'create_table({})', 'drop_table({})',
                'alter_table({}, {})', 'truncate_table({})', 'query_complexity({})',
                'estimated_rows({})', 'execution_time_ms({})', 'query_cost({})',
                'uses_index({}, {})', 'full_table_scan({})', 'nested_loop_join({}, {})',
                'hash_join({}, {})', 'sort_merge_join({}, {})', 'query_plan_node({}, {})',
                'query_optimizer_hint({})', 'parallel_execution({})', 'distributed_query({})',
                'cached_result({})', 'query_timeout({})', 'query_memory_usage({})',
                'temporary_table_created({})', 'result_set_size({})', 'affected_rows({})',
                'transaction_isolation_level({})', 'lock_acquired({}, {})', 'deadlock_detected({})'
            ],
            'constraint': [
                'primary_key_constraint({}, {})', 'foreign_key_constraint({}, {}, {}, {})',
                'unique_constraint({}, {})', 'not_null_constraint({}, {})', 'check_constraint({}, {}, {})',
                'default_constraint({}, {}, {})', 'exclusion_constraint({}, {}, {})',
                'constraint_violation({}, {})', 'constraint_valid({}, {})', 'constraint_enabled({}, {})',
                'constraint_deferred({}, {})', 'referential_integrity({}, {}, {}, {})',
                'cascade_delete({}, {}, {}, {})', 'cascade_update({}, {}, {}, {})',
                'restrict_delete({}, {}, {}, {})', 'restrict_update({}, {}, {}, {})',
                'domain_constraint({}, {}, {})', 'row_level_security({}, {})', 'column_level_security({}, {}, {})',
                'data_masking({}, {}, {})', 'encryption_constraint({}, {}, {})', 'compression_constraint({}, {})',
                'partition_constraint({}, {}, {})', 'constraint_cost({}, {})', 'constraint_selectivity({}, {})',
                'constraint_cardinality({}, {})', 'constraint_distribution({}, {})', 'constraint_correlation({}, {}, {})',
                'constraint_dependency({}, {}, {})', 'constraint_hierarchy({}, {}, {})'
            ],
            'semantic': [
                'entity_type({}, {})', 'relationship_type({}, {}, {})', 'inheritance({}, {})',
                'composition({}, {})', 'aggregation({}, {})', 'one_to_one({}, {})',
                'one_to_many({}, {})', 'many_to_many({}, {})', 'entity_attribute({}, {}, {})',
                'weak_entity({})', 'identifying_relationship({}, {})', 'non_identifying_relationship({}, {})',
                'business_rule({}, {})', 'domain_concept({})', 'semantic_type({}, {})',
                'data_lineage({}, {})', 'data_quality_rule({}, {})', 'master_data({})',
                'reference_data({})', 'transactional_data({})', 'analytical_data({})',
                'temporal_data({}, {})', 'versioned_data({}, {})', 'audit_trail({}, {})',
                'data_classification({}, {})', 'sensitivity_level({}, {})', 'retention_policy({}, {})',
                'archival_policy({}, {})', 'backup_policy({}, {})', 'recovery_policy({}, {})',
                'data_ownership({}, {})', 'data_stewardship({}, {})', 'data_governance_rule({}, {})',
                'compliance_requirement({}, {})', 'regulatory_constraint({}, {})', 'privacy_constraint({}, {})',
                'consent_management({}, {})', 'data_anonymization({}, {})', 'data_pseudonymization({}, {})',
                'cross_reference({}, {}, {})', 'synonym({}, {})', 'hierarchy_level({}, {}, {})'
            ]
        }
    
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract semantic facts from hidden states"""
        fact_predictions = {}
        
        for fact_type, classifier in self.fact_classifiers.items():
            predictions = classifier(hidden_states)
            fact_predictions[fact_type] = predictions
        
        return fact_predictions
    
    def decode_facts(self, predictions: Dict[str, torch.Tensor], 
                    threshold: float = 0.5) -> List[ExtractedFact]:
        """Decode predictions into structured facts"""
        facts = []
        
        for fact_type_str, preds in predictions.items():
            fact_type = FactType(fact_type_str)
            templates = self.fact_templates[fact_type_str]
            
            # Find predictions above threshold
            high_conf_indices = torch.where(preds > threshold)
            
            for batch_idx, seq_idx, template_idx in zip(*high_conf_indices):
                if template_idx.item() < len(templates):
                    template = templates[template_idx.item()]
                    confidence = preds[batch_idx, seq_idx, template_idx].item()
                    
                    # Create fact with placeholder arguments
                    fact_string = template.format(*['ARG'] * template.count('{}'))
                    
                    facts.append(ExtractedFact(
                        fact_string=fact_string,
                        fact_type=fact_type,
                        confidence=confidence,
                        source_tokens=[seq_idx.item()],
                        attributes={'template_idx': template_idx.item()}
                    ))
        
        return facts


class FactExtractor(nn.Module):
    """
    Comprehensive Fact Extractor
    
    Combines pattern matching and neural approaches to extract
    structured logical facts from neural representations.
    """
    
    def __init__(self, hidden_dim: int = 4096, extraction_threshold: float = 0.5,
                 max_facts_per_query: int = 100, enable_pattern_matching: bool = True):
        """
        Initialize fact extractor
        
        Args:
            hidden_dim: Hidden dimension of neural representations
            extraction_threshold: Minimum confidence for fact extraction
            max_facts_per_query: Maximum facts to extract per query
            enable_pattern_matching: Whether to use pattern matching
        """
        super().__init__()
        
        self.logger = logging.getLogger(__name__)
        self.hidden_dim = hidden_dim
        self.extraction_threshold = extraction_threshold
        self.max_facts_per_query = max_facts_per_query
        self.enable_pattern_matching = enable_pattern_matching
        
        # Neural semantic extractor
        self.semantic_extractor = SemanticFactExtractor(hidden_dim)
        
        # Pattern matcher
        if enable_pattern_matching:
            self.pattern_matcher = SQLPatternMatcher()
        else:
            self.pattern_matcher = None
        
        # Fact fusion network
        self.fact_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Fact ranking network
        self.fact_ranker = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),  # +1 for confidence
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.logger.info(f"FactExtractor initialized: threshold={extraction_threshold}, max_facts={max_facts_per_query}")
    
    def forward(self, hidden_states: torch.Tensor, 
                input_text: Optional[str] = None,
                symbolic_embeddings: Optional[torch.Tensor] = None) -> FactExtractionResult:
        """
        Extract facts from neural representations
        
        Args:
            hidden_states: Neural hidden states [batch_size, seq_len, hidden_dim]
            input_text: Original input text (optional, for pattern matching)
            symbolic_embeddings: Symbolic embeddings (optional)
            
        Returns:
            FactExtractionResult with extracted facts
        """
        import time
        start_time = time.time()
        
        batch_size, seq_len, _ = hidden_states.shape
        all_facts = []
        
        # Neural fact extraction
        semantic_predictions = self.semantic_extractor(hidden_states)
        neural_facts = self.semantic_extractor.decode_facts(
            semantic_predictions, 
            threshold=self.extraction_threshold
        )
        all_facts.extend(neural_facts)
        
        # Pattern-based fact extraction
        pattern_facts = []
        if self.pattern_matcher and input_text:
            pattern_facts = self.pattern_matcher.extract_sql_facts(input_text)
            all_facts.extend(pattern_facts)
        
        # Symbolic fact extraction (if symbolic embeddings provided)
        symbolic_facts = []
        if symbolic_embeddings is not None:
            symbolic_facts = self._extract_symbolic_facts(symbolic_embeddings)
            all_facts.extend(symbolic_facts)
        
        # Rank and filter facts
        ranked_facts = self._rank_and_filter_facts(all_facts, hidden_states)
        
        # Limit number of facts
        if len(ranked_facts) > self.max_facts_per_query:
            ranked_facts = ranked_facts[:self.max_facts_per_query]
        
        # Calculate statistics
        fact_counts = self._count_facts_by_type(ranked_facts)
        extraction_confidence = self._calculate_extraction_confidence(ranked_facts)
        
        processing_time = time.time() - start_time
        
        return FactExtractionResult(
            facts=ranked_facts,
            fact_counts=fact_counts,
            extraction_confidence=extraction_confidence,
            processing_time=processing_time,
            metadata={
                'neural_facts': len(neural_facts),
                'pattern_facts': len(pattern_facts) if self.pattern_matcher and input_text else 0,
                'symbolic_facts': len(symbolic_facts) if symbolic_embeddings is not None else 0,
                'total_candidates': len(all_facts),
                'final_facts': len(ranked_facts)
            }
        )
    
    def _extract_symbolic_facts(self, symbolic_embeddings: torch.Tensor) -> List[ExtractedFact]:
        """Extract facts from symbolic embeddings"""
        facts = []
        batch_size, seq_len, symbolic_dim = symbolic_embeddings.shape
        
        # Use fusion network to identify fact-rich positions
        fact_scores = self.fact_fusion(symbolic_embeddings)  # [batch_size, seq_len, 1]
        fact_scores = fact_scores.squeeze(-1)  # [batch_size, seq_len]
        
        # Extract facts from high-scoring positions
        for b in range(batch_size):
            for s in range(seq_len):
                score = fact_scores[b, s].item()
                if score > self.extraction_threshold:
                    # Generate symbolic fact based on embedding
                    fact_string = f"symbolic_fact_{b}_{s}(confidence={score:.3f})"
                    
                    facts.append(ExtractedFact(
                        fact_string=fact_string,
                        fact_type=FactType.SEMANTIC_FACT,
                        confidence=score,
                        source_tokens=[s],
                        attributes={'batch_idx': b, 'position': s}
                    ))
        
        return facts
    
    def _rank_and_filter_facts(self, facts: List[ExtractedFact], 
                              hidden_states: torch.Tensor) -> List[ExtractedFact]:
        """Rank and filter extracted facts"""
        if not facts:
            return facts
        
        # Sort by confidence (descending)
        facts.sort(key=lambda f: f.confidence, reverse=True)
        
        # Remove duplicates based on fact string
        seen_facts = set()
        unique_facts = []
        
        for fact in facts:
            if fact.fact_string not in seen_facts:
                seen_facts.add(fact.fact_string)
                unique_facts.append(fact)
        
        # Additional neural ranking (optional enhancement)
        if len(unique_facts) > self.max_facts_per_query:
            ranked_facts = self._neural_ranking(unique_facts, hidden_states)
            return ranked_facts[:self.max_facts_per_query]
        
        return unique_facts
    
    def _neural_ranking(self, facts: List[ExtractedFact], 
                       hidden_states: torch.Tensor) -> List[ExtractedFact]:
        """Use neural network to rank facts"""
        # Simple implementation: use existing confidence scores
        return sorted(facts, key=lambda f: f.confidence, reverse=True)
    
    def _count_facts_by_type(self, facts: List[ExtractedFact]) -> Dict[FactType, int]:
        """Count facts by type"""
        counts = {fact_type: 0 for fact_type in FactType}
        
        for fact in facts:
            counts[fact.fact_type] += 1
        
        return counts
    
    def _calculate_extraction_confidence(self, facts: List[ExtractedFact]) -> float:
        """Calculate overall extraction confidence"""
        if not facts:
            return 0.0
        
        confidences = [fact.confidence for fact in facts]
        return sum(confidences) / len(confidences)
    
    def extract_facts_from_text(self, text: str) -> List[ExtractedFact]:
        """Extract facts directly from text using pattern matching"""
        if not self.pattern_matcher:
            return []
        
        return self.pattern_matcher.extract_sql_facts(text)
    
    def get_fact_statistics(self, facts: List[ExtractedFact]) -> Dict[str, Any]:
        """Get detailed statistics about extracted facts"""
        if not facts:
            return {'total_facts': 0}
        
        stats = {
            'total_facts': len(facts),
            'average_confidence': sum(f.confidence for f in facts) / len(facts),
            'confidence_distribution': {
                'high_confidence': len([f for f in facts if f.confidence > 0.8]),
                'medium_confidence': len([f for f in facts if 0.5 < f.confidence <= 0.8]),
                'low_confidence': len([f for f in facts if f.confidence <= 0.5])
            },
            'fact_types': {ft.value: sum(1 for f in facts if f.fact_type == ft) for ft in FactType},
            'top_facts': [f.fact_string for f in sorted(facts, key=lambda x: x.confidence, reverse=True)[:10]]
        }
        
        return stats
    
    def export_facts_to_pyreason(self, facts: List[ExtractedFact]) -> List[str]:
        """Export facts in PyReason format"""
        pyreason_facts = []
        
        for fact in facts:
            # Convert fact string to PyReason format
            fact_str = fact.fact_string
            
            # Add confidence as a separate fact if needed
            if fact.confidence > 0.5:
                pyreason_facts.append(fact_str)
        
        return pyreason_facts
    
    def validate_extracted_facts(self, facts: List[ExtractedFact]) -> Dict[str, Any]:
        """Validate logical consistency of extracted facts"""
        validation_result = {
            'is_consistent': True,
            'inconsistencies': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check for basic inconsistencies
        fact_strings = [f.fact_string for f in facts]
        
        # Check for contradictory facts (simple heuristic)
        for i, fact1 in enumerate(fact_strings):
            for j, fact2 in enumerate(fact_strings[i+1:], i+1):
                if self._are_contradictory(fact1, fact2):
                    validation_result['inconsistencies'].append({
                        'fact1': fact1,
                        'fact2': fact2,
                        'reason': 'Potential contradiction detected'
                    })
                    validation_result['is_consistent'] = False
        
        # Check for missing important facts
        schema_facts = [f for f in facts if f.fact_type == FactType.SCHEMA_FACT]
        query_facts = [f for f in facts if f.fact_type == FactType.QUERY_FACT]
        
        if query_facts and not schema_facts:
            validation_result['warnings'].append(
                "Query facts found but no schema facts - consider adding schema information"
            )
        
        return validation_result
    
    def _are_contradictory(self, fact1: str, fact2: str) -> bool:
        """Simple heuristic to detect contradictory facts"""
        # This is a simple implementation - could be enhanced
        if 'table_exists' in fact1 and 'table_not_exists' in fact2:
            return fact1.split('(')[1] == fact2.split('(')[1]  # Same table
        
        return False


# Convenience functions
def create_fact_extractor(hidden_dim: int = 4096, **kwargs) -> FactExtractor:
    """Create a fact extractor with standard configuration"""
    return FactExtractor(hidden_dim, **kwargs)


def create_llama_fact_extractor(**kwargs) -> FactExtractor:
    """Create fact extractor optimized for Llama models"""
    defaults = {
        'hidden_dim': 4096,  # Llama hidden size
        'extraction_threshold': 0.5,
        'max_facts_per_query': 50,
        'enable_pattern_matching': True
    }
    defaults.update(kwargs)
    return FactExtractor(**defaults)


def extract_facts_from_sql(sql_query: str) -> List[ExtractedFact]:
    """Convenience function to extract facts from SQL query"""
    pattern_matcher = SQLPatternMatcher()
    return pattern_matcher.extract_sql_facts(sql_query)