#!/usr/bin/env python3
"""
Integration Evaluator for Fine-Tuning Pipeline

This module handles integration with the fine-tuning pipeline project,
enabling evaluation of real fine-tuned models with neurosymbolic enhancements.
"""

import sys
import os
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import importlib.util

import torch
import yaml
import numpy as np

@dataclass
class IntegrationResult:
    """Result container for integration evaluation"""
    
    compatibility_score: float
    base_model_performance: Dict[str, float]
    enhanced_model_performance: Dict[str, float]
    improvement_metrics: Dict[str, float]
    integration_issues: List[str]
    recommendations: List[str]

class IntegrationEvaluator:
    """
    Evaluates integration between neurosymbolic adapter and fine-tuned models
    
    This evaluator:
    1. Loads fine-tuned models from the fine-tuning pipeline
    2. Wraps them with neurosymbolic adapters
    3. Compares performance before and after enhancement
    4. Validates compatibility and integration quality
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize integration evaluator
        
        Args:
            config: Integration configuration including:
                - fine_tuning_project_path: Path to fine-tuning project
                - model_paths: Dictionary of model paths to evaluate
                - enable_comparative_analysis: Whether to compare models
                - enable_ablation_studies: Whether to run ablation studies
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Paths and configuration
        self.ft_project_path = Path(config.get('fine_tuning_project_path', '/Users/saptak/code/fine-tuning-small-llms'))
        self.model_paths = config.get('model_paths', {})
        
        # Integration settings
        self.enable_comparative = config.get('enable_comparative_analysis', True)
        self.enable_ablation = config.get('enable_ablation_studies', True)
        
        # Validate integration setup
        self._validate_integration_setup()
        
        self.logger.info(f"Integration evaluator initialized: {self.ft_project_path}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for integration evaluator"""
        logger = logging.getLogger("integration_evaluator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _validate_integration_setup(self):
        """Validate that integration can be performed"""
        
        integration_issues = []
        
        # Check fine-tuning project exists
        if not self.ft_project_path.exists():
            integration_issues.append(f"Fine-tuning project not found: {self.ft_project_path}")
        
        # Check for key components
        required_components = [
            "part3-training/src/fine_tune_model.py",
            "part3-training/configs/sql_expert.yaml",
            "data/datasets/sql_dataset_alpaca.json"
        ]
        
        for component in required_components:
            component_path = self.ft_project_path / component
            if not component_path.exists():
                integration_issues.append(f"Required component missing: {component}")
        
        # Check model paths
        for model_name, model_path in self.model_paths.items():
            full_path = self.ft_project_path / model_path
            if not full_path.exists():
                self.logger.warning(f"Model path not found (will create if needed): {full_path}")
        
        if integration_issues:
            self.logger.warning(f"Integration issues found: {integration_issues}")
        
        self.integration_issues = integration_issues
    
    def evaluate_integration(self, model: Any, test_datasets: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate integration quality between neurosymbolic adapter and fine-tuned models
        
        Args:
            model: Neurosymbolic model to evaluate
            test_datasets: Test datasets for evaluation
            
        Returns:
            Dictionary of integration metrics
        """
        
        self.logger.info("Starting integration evaluation")
        
        try:
            # Load fine-tuned models for comparison
            comparison_models = self._load_comparison_models()
            
            # Run comparative evaluation
            if self.enable_comparative and comparison_models:
                comparative_results = self._run_comparative_evaluation(
                    model, comparison_models, test_datasets
                )
            else:
                comparative_results = {}
            
            # Run ablation studies
            if self.enable_ablation:
                ablation_results = self._run_ablation_studies(model, test_datasets)
            else:
                ablation_results = {}
            
            # Calculate integration metrics
            integration_metrics = self._calculate_integration_metrics(
                comparative_results, ablation_results
            )
            
            self.logger.info("Integration evaluation completed")
            
            return integration_metrics
            
        except Exception as e:
            self.logger.error(f"Integration evaluation failed: {e}")
            return {
                'compatibility_score': 0.0,
                'integration_success': False,
                'error': str(e)
            }
    
    def _load_comparison_models(self) -> Dict[str, Any]:
        """Load models for comparison from fine-tuning project"""
        
        comparison_models = {}
        
        for model_name, model_path in self.model_paths.items():
            try:
                full_path = self.ft_project_path / model_path
                
                if full_path.exists():
                    # Load model using mock implementation for now
                    # In real implementation, would load actual fine-tuned models
                    comparison_models[model_name] = MockFineTunedModel(
                        model_path=str(full_path),
                        model_name=model_name
                    )
                    self.logger.info(f"Loaded comparison model: {model_name}")
                else:
                    self.logger.warning(f"Model not found: {full_path}")
                    
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {e}")
        
        return comparison_models
    
    def _run_comparative_evaluation(
        self, 
        neurosymbolic_model: Any, 
        comparison_models: Dict[str, Any], 
        test_datasets: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Run comparative evaluation between models"""
        
        self.logger.info("Running comparative evaluation")
        
        results = {}
        
        # Evaluate neurosymbolic model
        ns_results = self._evaluate_model_on_datasets(neurosymbolic_model, test_datasets, "neurosymbolic")
        results['neurosymbolic'] = ns_results
        
        # Evaluate comparison models
        for model_name, model in comparison_models.items():
            model_results = self._evaluate_model_on_datasets(model, test_datasets, model_name)
            results[model_name] = model_results
        
        return results
    
    def _run_ablation_studies(self, model: Any, test_datasets: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Run ablation studies to understand component contributions"""
        
        self.logger.info("Running ablation studies")
        
        ablation_results = {}
        
        # Test without constraint validation
        if hasattr(model, 'disable_constraint_validation'):
            try:
                model.disable_constraint_validation()
                no_constraints_results = self._evaluate_model_on_datasets(
                    model, test_datasets, "no_constraints"
                )
                ablation_results['no_constraints'] = no_constraints_results
                model.enable_constraint_validation()  # Re-enable
            except Exception as e:
                self.logger.warning(f"Constraint ablation failed: {e}")
        
        # Test without confidence estimation
        if hasattr(model, 'disable_confidence_estimation'):
            try:
                model.disable_confidence_estimation()
                no_confidence_results = self._evaluate_model_on_datasets(
                    model, test_datasets, "no_confidence"
                )
                ablation_results['no_confidence'] = no_confidence_results
                model.enable_confidence_estimation()  # Re-enable
            except Exception as e:
                self.logger.warning(f"Confidence ablation failed: {e}")
        
        # Test without explanation generation
        if hasattr(model, 'disable_explanation_generation'):
            try:
                model.disable_explanation_generation()
                no_explanation_results = self._evaluate_model_on_datasets(
                    model, test_datasets, "no_explanation"
                )
                ablation_results['no_explanation'] = no_explanation_results
                model.enable_explanation_generation()  # Re-enable
            except Exception as e:
                self.logger.warning(f"Explanation ablation failed: {e}")
        
        return ablation_results
    
    def _evaluate_model_on_datasets(
        self, 
        model: Any, 
        test_datasets: List[Dict[str, Any]], 
        model_name: str
    ) -> Dict[str, float]:
        """Evaluate a single model on test datasets"""
        
        all_results = []
        
        for dataset in test_datasets:
            test_cases = dataset.get('test_cases', [])
            
            for test_case in test_cases:
                try:
                    # Generate SQL
                    instruction = test_case.get('instruction', '')
                    schema = test_case.get('schema', '')
                    
                    if hasattr(model, 'generate_sql'):
                        result = model.generate_sql(instruction, schema)
                        sql = result.sql if hasattr(result, 'sql') else result.get('sql', '')
                        confidence = result.confidence if hasattr(result, 'confidence') else result.get('confidence', 0.5)
                        is_valid = result.is_valid if hasattr(result, 'is_valid') else result.get('is_valid', True)
                    else:
                        # Mock model
                        sql = f"SELECT * FROM table WHERE column = '{instruction[:20]}';"
                        confidence = 0.5
                        is_valid = True
                    
                    # Simple scoring
                    expected_sql = test_case.get('expected_sql', '')
                    correctness_score = self._calculate_sql_similarity(sql, expected_sql)
                    
                    all_results.append({
                        'correctness_score': correctness_score,
                        'confidence': confidence,
                        'is_valid': is_valid,
                        'has_sql': len(sql.strip()) > 0
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error evaluating test case with {model_name}: {e}")
                    all_results.append({
                        'correctness_score': 0.0,
                        'confidence': 0.0,
                        'is_valid': False,
                        'has_sql': False
                    })
        
        # Aggregate results
        if all_results:
            return {
                'avg_correctness': np.mean([r['correctness_score'] for r in all_results]),
                'avg_confidence': np.mean([r['confidence'] for r in all_results]),
                'validity_rate': np.mean([r['is_valid'] for r in all_results]),
                'success_rate': np.mean([r['has_sql'] for r in all_results]),
                'total_cases': len(all_results)
            }
        else:
            return {
                'avg_correctness': 0.0,
                'avg_confidence': 0.0,
                'validity_rate': 0.0,
                'success_rate': 0.0,
                'total_cases': 0
            }
    
    def _calculate_sql_similarity(self, sql1: str, sql2: str) -> float:
        """Calculate similarity between two SQL queries"""
        
        if not sql1 or not sql2:
            return 0.0
        
        # Simple token-based similarity
        tokens1 = set(sql1.lower().split())
        tokens2 = set(sql2.lower().split())
        
        if not tokens1 and not tokens2:
            return 1.0
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_integration_metrics(
        self, 
        comparative_results: Dict[str, Dict[str, float]], 
        ablation_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate overall integration metrics"""
        
        metrics = {}
        
        # Compatibility score based on successful evaluation
        if comparative_results:
            ns_results = comparative_results.get('neurosymbolic', {})
            success_rate = ns_results.get('success_rate', 0.0)
            validity_rate = ns_results.get('validity_rate', 0.0)
            
            compatibility_score = (success_rate + validity_rate) / 2.0
            metrics['compatibility_score'] = compatibility_score
            
            # Performance comparison
            if len(comparative_results) > 1:
                ns_correctness = ns_results.get('avg_correctness', 0.0)
                
                # Compare with base models
                base_correctness_scores = []
                for model_name, results in comparative_results.items():
                    if model_name != 'neurosymbolic':
                        base_correctness_scores.append(results.get('avg_correctness', 0.0))
                
                if base_correctness_scores:
                    avg_base_correctness = np.mean(base_correctness_scores)
                    improvement = ns_correctness - avg_base_correctness
                    metrics['correctness_improvement'] = improvement
                    metrics['relative_improvement'] = improvement / avg_base_correctness if avg_base_correctness > 0 else 0.0
        
        # Ablation study insights
        if ablation_results and 'neurosymbolic' in comparative_results:
            full_performance = comparative_results['neurosymbolic'].get('avg_correctness', 0.0)
            
            # Constraint validation contribution
            if 'no_constraints' in ablation_results:
                no_constraints_perf = ablation_results['no_constraints'].get('avg_correctness', 0.0)
                constraint_contribution = full_performance - no_constraints_perf
                metrics['constraint_validation_contribution'] = constraint_contribution
            
            # Confidence estimation contribution
            if 'no_confidence' in ablation_results:
                no_confidence_perf = ablation_results['no_confidence'].get('avg_correctness', 0.0)
                confidence_contribution = full_performance - no_confidence_perf
                metrics['confidence_estimation_contribution'] = confidence_contribution
        
        # Integration success indicator
        metrics['integration_success'] = len(self.integration_issues) == 0
        metrics['integration_issues_count'] = len(self.integration_issues)
        
        return metrics
    
    def create_integration_test_dataset(self) -> List[Dict[str, Any]]:
        """Create test dataset specifically for integration testing"""
        
        # Load dataset from fine-tuning project if available
        dataset_path = self.ft_project_path / "data/datasets/sql_dataset_alpaca.json"
        
        if dataset_path.exists():
            try:
                with open(dataset_path, 'r') as f:
                    original_dataset = json.load(f)
                
                # Convert to evaluation format
                test_cases = []
                for item in original_dataset[:20]:  # Use first 20 for quick testing
                    test_cases.append({
                        'instruction': item.get('instruction', ''),
                        'schema': item.get('input', ''),
                        'expected_sql': item.get('output', ''),
                        'complexity': 'medium'
                    })
                
                return [{
                    'name': 'integration_test_dataset',
                    'description': 'Dataset from fine-tuning project for integration testing',
                    'test_cases': test_cases
                }]
                
            except Exception as e:
                self.logger.warning(f"Failed to load fine-tuning dataset: {e}")
        
        # Fallback to synthetic dataset
        return self._create_synthetic_integration_dataset()
    
    def _create_synthetic_integration_dataset(self) -> List[Dict[str, Any]]:
        """Create synthetic dataset for integration testing"""
        
        test_cases = [
            {
                'instruction': 'Find all active users',
                'schema': 'users (id, name, email, status)',
                'expected_sql': 'SELECT * FROM users WHERE status = "active";',
                'complexity': 'simple'
            },
            {
                'instruction': 'Count orders by customer with their names',
                'schema': 'customers (id, name), orders (id, customer_id, amount)',
                'expected_sql': 'SELECT c.name, COUNT(o.id) FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name;',
                'complexity': 'medium'
            },
            {
                'instruction': 'Find customers who have never placed an order',
                'schema': 'customers (id, name), orders (id, customer_id, amount)',
                'expected_sql': 'SELECT c.* FROM customers c LEFT JOIN orders o ON c.id = o.customer_id WHERE o.customer_id IS NULL;',
                'complexity': 'medium'
            },
            {
                'instruction': 'Get top 5 customers by total order value',
                'schema': 'customers (id, name), orders (id, customer_id, amount)',
                'expected_sql': 'SELECT c.name, SUM(o.amount) as total FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name ORDER BY total DESC LIMIT 5;',
                'complexity': 'complex'
            }
        ]
        
        return [{
            'name': 'synthetic_integration_dataset',
            'description': 'Synthetic dataset for integration testing',
            'test_cases': test_cases
        }]

class MockFineTunedModel:
    """Mock fine-tuned model for testing integration"""
    
    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
    
    def generate_sql(self, instruction: str, schema: str = "") -> Dict[str, Any]:
        """Mock SQL generation"""
        
        # Simple rule-based generation for testing
        instruction_lower = instruction.lower()
        
        if 'active users' in instruction_lower:
            sql = 'SELECT * FROM users WHERE status = "active";'
        elif 'count' in instruction_lower and 'customer' in instruction_lower:
            sql = 'SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id;'
        elif 'never' in instruction_lower and 'order' in instruction_lower:
            sql = 'SELECT c.* FROM customers c LEFT JOIN orders o ON c.id = o.customer_id WHERE o.customer_id IS NULL;'
        elif 'top' in instruction_lower:
            sql = 'SELECT c.name, SUM(o.amount) FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.name ORDER BY SUM(o.amount) DESC LIMIT 5;'
        else:
            sql = f'SELECT * FROM table WHERE column LIKE "%{instruction[:20]}%";'
        
        return {
            'sql': sql,
            'confidence': 0.8,
            'is_valid': True,
            'violations': [],
            'explanation': f'Generated by {self.model_name} for: {instruction}'
        }