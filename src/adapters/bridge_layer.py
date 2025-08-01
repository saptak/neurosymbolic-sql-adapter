#!/usr/bin/env python3
"""
Bridge Layer

Neural-symbolic translation layer that converts neural representations
to symbolic reasoning contexts and vice versa.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import math


@dataclass
class SymbolicContext:
    """Container for symbolic reasoning context"""
    embeddings: torch.Tensor  # Symbolic space embeddings
    facts: Optional[List[str]] = None  # Extracted logical facts
    attention_weights: Optional[torch.Tensor] = None  # Attention over symbolic elements
    reasoning_trace: Optional[List[str]] = None  # Step-by-step reasoning
    confidence_mask: Optional[torch.Tensor] = None  # Confidence for each symbolic element


class MultiHeadSymbolicAttention(nn.Module):
    """Multi-head attention specialized for symbolic reasoning"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, embed_dim = query.size()
        
        # Project to query, key, value
        Q = self.query_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if attn_mask is not None:
            scores.masked_fill_(attn_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Final projection
        output = self.out_proj(context)
        
        return output, attn_weights.mean(dim=1)  # Average attention over heads


class SymbolicTransformerBlock(nn.Module):
    """Transformer block specialized for symbolic reasoning"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, ff_dim: Optional[int] = None,
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        
        self.embed_dim = embed_dim
        ff_dim = ff_dim or 4 * embed_dim
        
        # Multi-head attention
        self.attention = MultiHeadSymbolicAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Symbolic reasoning gates
        self.symbolic_gate = nn.Linear(embed_dim, embed_dim)
        self.reasoning_gate = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual connection
        attn_out, attn_weights = self.attention(x, x, x, attn_mask)
        x = self.norm1(x + attn_out)
        
        # Apply symbolic reasoning gates
        symbolic_signal = torch.sigmoid(self.symbolic_gate(x))
        reasoning_signal = torch.sigmoid(self.reasoning_gate(x))
        x = x * symbolic_signal * reasoning_signal
        
        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x, attn_weights


class BridgeLayer(nn.Module):
    """
    Bridge Layer for Neural-Symbolic Translation
    
    Converts neural language model representations to symbolic reasoning
    contexts that can be processed by the PyReason engine.
    """
    
    def __init__(self, neural_dim: int = 4096, symbolic_dim: int = 512, 
                 bridge_dim: int = 256, num_layers: int = 2, num_heads: int = 8,
                 dropout: float = 0.1, max_facts: int = 50):
        """
        Initialize bridge layer
        
        Args:
            neural_dim: Dimension of neural representations (e.g., 4096 for Llama)
            symbolic_dim: Dimension of symbolic space
            bridge_dim: Intermediate bridge dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            max_facts: Maximum number of facts to extract
        """
        super().__init__()
        
        self.logger = logging.getLogger(__name__)
        self.neural_dim = neural_dim
        self.symbolic_dim = symbolic_dim
        self.bridge_dim = bridge_dim
        self.max_facts = max_facts
        
        # Neural to symbolic projection
        self.neural_to_bridge = nn.Sequential(
            nn.Linear(neural_dim, bridge_dim),
            nn.LayerNorm(bridge_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Symbolic transformer layers
        self.transformer_layers = nn.ModuleList([
            SymbolicTransformerBlock(bridge_dim, num_heads, bridge_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Bridge to symbolic projection
        self.bridge_to_symbolic = nn.Sequential(
            nn.Linear(bridge_dim, symbolic_dim),
            nn.LayerNorm(symbolic_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Fact extraction heads
        self.fact_classifier = nn.Sequential(
            nn.Linear(symbolic_dim, symbolic_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(symbolic_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Symbolic concept embeddings (learned representations for SQL concepts)
        self.concept_embeddings = nn.Embedding(100, symbolic_dim)  # 100 SQL concepts
        self.concept_names = self._initialize_concept_names()
        
        # Position encoding for symbolic elements
        self.positional_encoding = self._create_positional_encoding(max_facts, symbolic_dim)
        
        self.logger.info(f"BridgeLayer initialized: {neural_dim} -> {bridge_dim} -> {symbolic_dim}")
    
    def _initialize_concept_names(self) -> List[str]:
        """Initialize symbolic concept names"""
        return [
            # SQL Operations
            "SELECT", "FROM", "WHERE", "JOIN", "GROUP_BY", "ORDER_BY", "HAVING",
            "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER",
            
            # Aggregate Functions
            "COUNT", "SUM", "AVG", "MIN", "MAX", "STDDEV", "VARIANCE",
            
            # Comparison Operators
            "EQUALS", "NOT_EQUALS", "LESS_THAN", "GREATER_THAN", "LIKE", "IN", "EXISTS",
            
            # Logical Operators
            "AND", "OR", "NOT", "BETWEEN", "IS_NULL", "IS_NOT_NULL",
            
            # Data Types
            "INTEGER", "VARCHAR", "TEXT", "DATE", "TIMESTAMP", "BOOLEAN", "DECIMAL",
            
            # Constraints
            "PRIMARY_KEY", "FOREIGN_KEY", "UNIQUE", "NOT_NULL", "CHECK",
            
            # Table Operations
            "INNER_JOIN", "LEFT_JOIN", "RIGHT_JOIN", "FULL_JOIN", "CROSS_JOIN",
            
            # Schema Elements
            "TABLE", "COLUMN", "INDEX", "VIEW", "TRIGGER", "PROCEDURE", "FUNCTION",
            
            # Semantic Relationships
            "ONE_TO_MANY", "MANY_TO_ONE", "MANY_TO_MANY", "INHERITANCE", "COMPOSITION",
            
            # Query Patterns
            "SUBQUERY", "WINDOW_FUNCTION", "RECURSIVE", "UNION", "INTERSECT", "EXCEPT",
            
            # Validation Concepts
            "VALID_SYNTAX", "SEMANTIC_ERROR", "CONSTRAINT_VIOLATION", "TYPE_MISMATCH",
            
            # Placeholder concepts for extension
            *[f"CONCEPT_{i}" for i in range(64, 100)]
        ]
    
    def _create_positional_encoding(self, max_len: int, dim: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, dim, 2).float() * 
                           -(math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Add batch dimension
    
    def forward(self, neural_representations: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> SymbolicContext:
        """
        Forward pass: Neural -> Symbolic translation
        
        Args:
            neural_representations: Neural embeddings [batch_size, seq_len, neural_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            SymbolicContext with symbolic representations and extracted facts
        """
        batch_size, seq_len, _ = neural_representations.shape
        
        # Project neural representations to bridge space
        bridge_repr = self.neural_to_bridge(neural_representations)
        
        # Apply transformer layers for symbolic reasoning
        symbolic_attention_weights = []
        x = bridge_repr
        
        for layer in self.transformer_layers:
            x, attn_weights = layer(x, attention_mask)
            symbolic_attention_weights.append(attn_weights)
        
        # Project to symbolic space
        symbolic_embeddings = self.bridge_to_symbolic(x)
        
        # Add positional encoding
        if symbolic_embeddings.size(1) <= self.positional_encoding.size(1):
            pos_encoding = self.positional_encoding[:, :symbolic_embeddings.size(1), :].to(symbolic_embeddings.device)
            symbolic_embeddings = symbolic_embeddings + pos_encoding
        
        # Extract symbolic facts
        fact_scores = self.fact_classifier(symbolic_embeddings).squeeze(-1)  # [batch_size, seq_len]
        
        # Generate confidence mask
        confidence_mask = fact_scores > 0.5
        
        # Extract top-k facts per batch
        extracted_facts = self._extract_facts(symbolic_embeddings, fact_scores, batch_size)
        
        # Compute attention over symbolic concepts
        concept_embeddings = self.concept_embeddings.weight  # [num_concepts, symbolic_dim]
        concept_attention = torch.matmul(
            symbolic_embeddings,  # [batch_size, seq_len, symbolic_dim]
            concept_embeddings.t()  # [symbolic_dim, num_concepts]
        )  # [batch_size, seq_len, num_concepts]
        
        concept_attention = F.softmax(concept_attention, dim=-1)
        
        return SymbolicContext(
            embeddings=symbolic_embeddings,
            facts=extracted_facts,
            attention_weights=torch.stack(symbolic_attention_weights, dim=1),  # [batch_size, num_layers, seq_len, seq_len]
            reasoning_trace=self._generate_reasoning_trace(concept_attention, fact_scores),
            confidence_mask=confidence_mask
        )
    
    def _extract_facts(self, symbolic_embeddings: torch.Tensor, fact_scores: torch.Tensor, 
                      batch_size: int) -> List[List[str]]:
        """Extract symbolic facts from representations"""
        extracted_facts = []
        
        for b in range(batch_size):
            batch_facts = []
            scores = fact_scores[b]
            embeddings = symbolic_embeddings[b]
            
            # Get top-k scoring positions
            top_k = min(self.max_facts, len(scores))
            top_indices = torch.topk(scores, top_k, dim=0).indices
            
            for idx in top_indices:
                if scores[idx] > 0.3:  # Threshold for fact extraction
                    # Find closest concept
                    embedding = embeddings[idx]
                    concept_similarities = F.cosine_similarity(
                        embedding.unsqueeze(0),
                        self.concept_embeddings.weight,
                        dim=1
                    )
                    
                    best_concept_idx = torch.argmax(concept_similarities).item()
                    concept_name = self.concept_names[best_concept_idx]
                    confidence = scores[idx].item()
                    
                    fact = f"{concept_name.lower()}(confidence={confidence:.3f})"
                    batch_facts.append(fact)
            
            extracted_facts.append(batch_facts)
        
        return extracted_facts
    
    def _generate_reasoning_trace(self, concept_attention: torch.Tensor, 
                                fact_scores: torch.Tensor) -> List[List[str]]:
        """Generate reasoning trace from attention patterns"""
        batch_size = concept_attention.size(0)
        reasoning_traces = []
        
        for b in range(batch_size):
            trace = []
            attention = concept_attention[b]  # [seq_len, num_concepts]
            scores = fact_scores[b]  # [seq_len]
            
            # Find positions with high fact scores
            high_score_positions = torch.where(scores > 0.5)[0]
            
            for pos in high_score_positions[:10]:  # Top 10 reasoning steps
                pos_attention = attention[pos]
                top_concepts = torch.topk(pos_attention, 3).indices
                
                concept_names = [self.concept_names[idx.item()] for idx in top_concepts]
                step = f"Position {pos}: Focused on {', '.join(concept_names)}"
                trace.append(step)
            
            reasoning_traces.append(trace)
        
        return reasoning_traces
    
    def symbolic_to_neural(self, symbolic_context: SymbolicContext) -> torch.Tensor:
        """
        Reverse translation: Symbolic -> Neural
        
        Args:
            symbolic_context: Symbolic reasoning context
            
        Returns:
            Neural representations
        """
        symbolic_embeddings = symbolic_context.embeddings
        
        # Project back through the bridge
        bridge_repr = torch.relu(
            torch.matmul(symbolic_embeddings, self.bridge_to_symbolic[0].weight.t()) +
            self.bridge_to_symbolic[0].bias
        )
        
        # Project to neural space
        neural_repr = torch.relu(
            torch.matmul(bridge_repr, self.neural_to_bridge[0].weight.t()) +
            self.neural_to_bridge[0].bias
        )
        
        return neural_repr
    
    def get_concept_similarities(self, query_embedding: torch.Tensor) -> Dict[str, float]:
        """Get similarities between query and SQL concepts"""
        concept_embeddings = self.concept_embeddings.weight
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            concept_embeddings,
            dim=1
        )
        
        concept_similarities = {}
        # Ensure we don't exceed available concepts
        num_concepts = min(len(self.concept_names), similarities.size(0))
        for i in range(num_concepts):
            concept_name = self.concept_names[i]
            concept_similarities[concept_name] = similarities[i].item()
        
        return concept_similarities
    
    def explain_symbolic_transformation(self, neural_input: torch.Tensor) -> Dict[str, Any]:
        """Provide detailed explanation of neural-symbolic transformation"""
        with torch.no_grad():
            symbolic_context = self.forward(neural_input)
            
            explanation = {
                'input_shape': list(neural_input.shape),
                'symbolic_shape': list(symbolic_context.embeddings.shape),
                'extracted_facts_count': len(symbolic_context.facts[0]) if symbolic_context.facts else 0,
                'top_concepts': {},
                'reasoning_steps': symbolic_context.reasoning_trace[0] if symbolic_context.reasoning_trace else [],
                'confidence_distribution': {}
            }
            
            # Analyze concept activation
            if symbolic_context.embeddings.size(0) > 0:
                avg_embedding = symbolic_context.embeddings[0].mean(dim=0)
                concept_similarities = self.get_concept_similarities(avg_embedding)
                
                # Get top 10 activated concepts
                sorted_concepts = sorted(concept_similarities.items(), key=lambda x: x[1], reverse=True)
                explanation['top_concepts'] = dict(sorted_concepts[:10])
            
            # Analyze confidence distribution
            if symbolic_context.confidence_mask is not None:
                confidence_mask = symbolic_context.confidence_mask[0]  # First batch
                explanation['confidence_distribution'] = {
                    'high_confidence_positions': int(confidence_mask.sum().item()),
                    'total_positions': int(confidence_mask.numel()),
                    'confidence_ratio': float(confidence_mask.float().mean().item())
                }
            
            return explanation
    
    def get_parameters_count(self) -> Dict[str, int]:
        """Get parameter count breakdown"""
        total_params = sum(p.numel() for p in self.parameters())
        
        component_params = {
            'neural_to_bridge': sum(p.numel() for p in self.neural_to_bridge.parameters()),
            'transformer_layers': sum(p.numel() for p in self.transformer_layers.parameters()),
            'bridge_to_symbolic': sum(p.numel() for p in self.bridge_to_symbolic.parameters()),
            'fact_classifier': sum(p.numel() for p in self.fact_classifier.parameters()),
            'concept_embeddings': sum(p.numel() for p in self.concept_embeddings.parameters()),
            'total': total_params
        }
        
        return component_params


# Convenience functions
def create_bridge_layer(neural_dim: int = 4096, symbolic_dim: int = 512, **kwargs) -> BridgeLayer:
    """Create a new bridge layer with standard configuration"""
    return BridgeLayer(neural_dim, symbolic_dim, **kwargs)


def create_llama_bridge() -> BridgeLayer:
    """Create bridge layer optimized for Llama models"""
    return BridgeLayer(
        neural_dim=4096,  # Llama hidden size
        symbolic_dim=512,
        bridge_dim=256,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
        max_facts=50
    )