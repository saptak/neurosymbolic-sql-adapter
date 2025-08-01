#!/usr/bin/env python3
"""
Comprehensive Evaluation Framework for Neurosymbolic SQL Adapter

This framework provides end-to-end evaluation capabilities including:
- Integration with fine-tuned models
- SQL correctness assessment  
- Constraint satisfaction validation
- Reasoning quality evaluation
- Performance benchmarking
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local imports
from .sql_metrics import SQLMetrics
from .reasoning_quality import ReasoningQualityAssessment
from .performance_benchmarks import PerformanceBenchmarks
from .integration_evaluator import IntegrationEvaluator

@dataclass
class EvaluationResult:
    """Comprehensive evaluation result container"""
    
    # Test metadata
    test_id: str
    timestamp: str
    model_info: Dict[str, Any]
    
    # SQL correctness metrics
    sql_correctness: Dict[str, float]
    
    # Constraint satisfaction metrics
    constraint_metrics: Dict[str, float]
    
    # Reasoning quality metrics
    reasoning_metrics: Dict[str, float]
    
    # Performance metrics
    performance_metrics: Dict[str, float]
    
    # Integration metrics (if applicable)
    integration_metrics: Optional[Dict[str, float]] = None
    
    # Detailed results
    detailed_results: Optional[List[Dict[str, Any]]] = None
    
    # Overall scores
    overall_score: Optional[float] = None
    recommendations: Optional[List[str]] = None

class EvaluationFramework:
    """
    Comprehensive evaluation framework for neurosymbolic SQL adapter
    
    Supports evaluation of:
    1. Mock models (for testing)
    2. Fine-tuned models from the fine-tuning pipeline
    3. Hybrid neurosymbolic models
    4. Comparative analysis between different approaches
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluation framework
        
        Args:
            config: Evaluation configuration including:
                - test_datasets: List of test datasets
                - metrics_config: Configuration for metrics
                - integration_config: Fine-tuning pipeline integration settings
                - output_dir: Directory for results
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize evaluation components
        self.sql_metrics = SQLMetrics(config.get('sql_metrics', {}))
        self.reasoning_quality = ReasoningQualityAssessment(config.get('reasoning_quality', {}))
        self.performance_benchmarks = PerformanceBenchmarks(config.get('performance', {}))
        
        # Integration evaluator for fine-tuning pipeline
        if config.get('enable_integration', False):
            self.integration_evaluator = IntegrationEvaluator(
                config.get('integration_config', {})
            )
        else:
            self.integration_evaluator = None
        
        # Results storage
        self.output_dir = Path(config.get('output_dir', './evaluation_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Evaluation framework initialized: {self.output_dir}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for evaluation framework"""
        logger = logging.getLogger(f"evaluation_framework")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def evaluate_model(
        self, 
        model: Any, 
        test_datasets: List[Dict[str, Any]],
        model_info: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Comprehensive evaluation of a neurosymbolic model
        
        Args:
            model: Model to evaluate (can be mock, fine-tuned, or hybrid)
            test_datasets: List of test datasets
            model_info: Optional model metadata
            
        Returns:
            EvaluationResult with comprehensive metrics
        """
        
        test_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"Starting comprehensive evaluation: {test_id}")
        
        # Collect model information
        if model_info is None:
            model_info = self._extract_model_info(model)
        
        # Initialize result containers
        all_sql_results = []
        all_constraint_results = []
        all_reasoning_results = []
        all_performance_results = []
        detailed_results = []
        
        # Process each test dataset
        for dataset_idx, dataset in enumerate(test_datasets):
            self.logger.info(f"Processing dataset {dataset_idx + 1}/{len(test_datasets)}: {dataset.get('name', 'unnamed')}")
            
            dataset_results = self._evaluate_dataset(model, dataset)
            
            # Aggregate results
            all_sql_results.extend(dataset_results['sql_results'])
            all_constraint_results.extend(dataset_results['constraint_results'])
            all_reasoning_results.extend(dataset_results['reasoning_results'])
            all_performance_results.extend(dataset_results['performance_results'])
            detailed_results.extend(dataset_results['detailed_results'])
        
        # Calculate aggregate metrics
        sql_correctness = self.sql_metrics.aggregate_results(all_sql_results)
        constraint_metrics = self._aggregate_constraint_metrics(all_constraint_results)
        reasoning_metrics = self.reasoning_quality.aggregate_results(all_reasoning_results)
        performance_metrics = self.performance_benchmarks.aggregate_results(all_performance_results)
        
        # Integration metrics (if applicable)
        integration_metrics = None
        if self.integration_evaluator and hasattr(model, 'base_model'):
            integration_metrics = self.integration_evaluator.evaluate_integration(
                model, test_datasets
            )
        
        # Calculate overall score
        overall_score = self._calculate_overall_score({
            'sql_correctness': sql_correctness,
            'constraint_metrics': constraint_metrics,
            'reasoning_metrics': reasoning_metrics,
            'performance_metrics': performance_metrics
        })
        
        # Generate recommendations
        recommendations = self._generate_recommendations({
            'sql_correctness': sql_correctness,
            'constraint_metrics': constraint_metrics,
            'reasoning_metrics': reasoning_metrics,
            'performance_metrics': performance_metrics,
            'integration_metrics': integration_metrics
        })
        
        # Create comprehensive result
        result = EvaluationResult(
            test_id=test_id,
            timestamp=datetime.now().isoformat(),
            model_info=model_info,
            sql_correctness=sql_correctness,
            constraint_metrics=constraint_metrics,
            reasoning_metrics=reasoning_metrics,
            performance_metrics=performance_metrics,
            integration_metrics=integration_metrics,
            detailed_results=detailed_results,
            overall_score=overall_score,
            recommendations=recommendations
        )
        
        # Save results
        self._save_results(result)
        
        self.logger.info(f"Evaluation completed: {test_id} (Score: {overall_score:.3f})")
        
        return result
    
    def _evaluate_dataset(self, model: Any, dataset: Dict[str, Any]) -> Dict[str, List]:
        """Evaluate model on a single dataset"""
        
        test_cases = dataset.get('test_cases', [])
        dataset_name = dataset.get('name', 'unnamed')
        
        sql_results = []
        constraint_results = []
        reasoning_results = []
        performance_results = []
        detailed_results = []
        
        for i, test_case in enumerate(tqdm(test_cases, desc=f"Evaluating {dataset_name}")):
            try:
                # Single test case evaluation
                case_result = self._evaluate_test_case(model, test_case, dataset_name, i)
                
                sql_results.append(case_result['sql_result'])
                constraint_results.append(case_result['constraint_result'])
                reasoning_results.append(case_result['reasoning_result'])
                performance_results.append(case_result['performance_result'])
                detailed_results.append(case_result['detailed_result'])
                
            except Exception as e:
                self.logger.error(f"Error evaluating test case {i}: {e}")
                # Add error case to results
                error_result = self._create_error_result(test_case, str(e), dataset_name, i)
                detailed_results.append(error_result)
        
        return {
            'sql_results': sql_results,
            'constraint_results': constraint_results,
            'reasoning_results': reasoning_results,
            'performance_results': performance_results,
            'detailed_results': detailed_results
        }
    
    def _evaluate_test_case(self, model: Any, test_case: Dict[str, Any], dataset_name: str, case_idx: int) -> Dict[str, Any]:
        """Evaluate a single test case"""
        
        instruction = test_case.get('instruction', '')
        schema = test_case.get('schema', '')
        expected_sql = test_case.get('expected_sql', '')
        expected_result = test_case.get('expected_result', None)
        
        # Performance timing
        start_time = time.time()
        
        # Generate SQL with model
        if hasattr(model, 'generate_sql'):
            # Neurosymbolic model
            generation_result = model.generate_sql(instruction, schema)
            generated_sql = generation_result.sql if hasattr(generation_result, 'sql') else generation_result.get('sql', '')
            confidence = generation_result.confidence if hasattr(generation_result, 'confidence') else generation_result.get('confidence', 0.0)
            is_valid = generation_result.is_valid if hasattr(generation_result, 'is_valid') else generation_result.get('is_valid', True)
            violations = generation_result.violations if hasattr(generation_result, 'violations') else generation_result.get('violations', [])
            explanation = generation_result.explanation if hasattr(generation_result, 'explanation') else generation_result.get('explanation', '')
        else:
            # Mock or basic model
            generated_sql = f"SELECT * FROM table WHERE condition = '{instruction[:20]}...'"
            confidence = 0.5
            is_valid = True
            violations = []
            explanation = "Mock explanation"
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Evaluate SQL correctness
        sql_result = self.sql_metrics.evaluate_sql(
            generated_sql=generated_sql,
            expected_sql=expected_sql,
            schema=schema
        )
        
        # Evaluate constraint satisfaction
        constraint_result = {
            'is_valid': is_valid,
            'violations_count': len(violations),
            'violations': violations,
            'confidence': confidence
        }
        
        # Evaluate reasoning quality
        reasoning_result = self.reasoning_quality.evaluate_reasoning(
            explanation=explanation,
            generated_sql=generated_sql,
            instruction=instruction,
            schema=schema
        )
        
        # Performance metrics
        performance_result = {
            'generation_time': generation_time,
            'sql_length': len(generated_sql),
            'complexity_score': self._calculate_sql_complexity(generated_sql)
        }
        
        # Detailed result for reporting
        detailed_result = {
            'dataset': dataset_name,
            'case_index': case_idx,
            'instruction': instruction,
            'schema': schema,
            'expected_sql': expected_sql,
            'generated_sql': generated_sql,
            'confidence': confidence,
            'is_valid': is_valid,
            'violations': violations,
            'explanation': explanation,
            'generation_time': generation_time,
            'sql_correctness_score': sql_result.get('overall_score', 0.0),
            'reasoning_quality_score': reasoning_result.get('overall_score', 0.0),
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            'sql_result': sql_result,
            'constraint_result': constraint_result,
            'reasoning_result': reasoning_result, 
            'performance_result': performance_result,
            'detailed_result': detailed_result
        }
    
    def _extract_model_info(self, model: Any) -> Dict[str, Any]:
        """Extract model information for metadata"""
        
        model_info = {
            'model_type': type(model).__name__,
            'has_neurosymbolic': hasattr(model, 'generate_sql'),
            'has_base_model': hasattr(model, 'base_model'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Try to extract additional info
        if hasattr(model, 'get_status'):
            try:
                status = model.get_status()
                model_info.update(status)
            except:
                pass
        
        if hasattr(model, 'config'):
            try:
                model_info['config'] = model.config
            except:
                pass
        
        return model_info
    
    def _aggregate_constraint_metrics(self, constraint_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate constraint satisfaction metrics"""
        
        if not constraint_results:
            return {}
        
        valid_count = sum(1 for r in constraint_results if r.get('is_valid', True))
        total_count = len(constraint_results)
        
        total_violations = sum(r.get('violations_count', 0) for r in constraint_results)
        avg_confidence = np.mean([r.get('confidence', 0.0) for r in constraint_results])
        
        return {
            'constraint_satisfaction_rate': valid_count / total_count if total_count > 0 else 0.0,
            'average_violations_per_query': total_violations / total_count if total_count > 0 else 0.0,
            'average_confidence': float(avg_confidence),
            'total_test_cases': total_count,
            'valid_queries': valid_count,
            'invalid_queries': total_count - valid_count
        }
    
    def _calculate_sql_complexity(self, sql: str) -> float:
        """Calculate SQL complexity score"""
        
        sql_lower = sql.lower()
        complexity_score = 0.0
        
        # Basic scoring
        complexity_score += sql_lower.count('select') * 1.0
        complexity_score += sql_lower.count('join') * 2.0
        complexity_score += sql_lower.count('inner join') * 2.0
        complexity_score += sql_lower.count('left join') * 2.5
        complexity_score += sql_lower.count('right join') * 2.5
        complexity_score += sql_lower.count('outer join') * 3.0
        complexity_score += sql_lower.count('group by') * 2.0
        complexity_score += sql_lower.count('order by') * 1.5
        complexity_score += sql_lower.count('having') * 2.5
        complexity_score += sql_lower.count('case when') * 3.0
        complexity_score += sql_lower.count('exists') * 3.0
        complexity_score += sql_lower.count('in (select') * 3.0
        
        return complexity_score
    
    def _calculate_overall_score(self, metrics: Dict[str, Dict[str, float]]) -> float:
        """Calculate weighted overall score"""
        
        weights = {
            'sql_correctness': 0.4,
            'constraint_metrics': 0.3,
            'reasoning_metrics': 0.2,
            'performance_metrics': 0.1
        }
        
        overall_score = 0.0
        total_weight = 0.0
        
        for metric_category, weight in weights.items():
            category_metrics = metrics.get(metric_category, {})
            
            if metric_category == 'sql_correctness':
                score = category_metrics.get('overall_score', 0.0)
            elif metric_category == 'constraint_metrics':
                score = category_metrics.get('constraint_satisfaction_rate', 0.0)
            elif metric_category == 'reasoning_metrics':
                score = category_metrics.get('overall_score', 0.0)
            elif metric_category == 'performance_metrics':
                # Normalize performance score (lower time = higher score)
                avg_time = category_metrics.get('average_generation_time', 1.0)
                score = max(0.0, 1.0 - min(avg_time / 5.0, 1.0))  # 5s = 0 score
            else:
                score = 0.0
            
            overall_score += score * weight
            total_weight += weight
        
        return overall_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_recommendations(self, metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        
        recommendations = []
        
        # SQL correctness recommendations
        sql_metrics = metrics.get('sql_correctness', {})
        if sql_metrics.get('syntax_accuracy', 1.0) < 0.9:
            recommendations.append("Improve SQL syntax accuracy through additional training data")
        
        if sql_metrics.get('semantic_correctness', 1.0) < 0.8:
            recommendations.append("Enhance semantic understanding with more diverse training examples")
        
        # Constraint satisfaction recommendations
        constraint_metrics = metrics.get('constraint_metrics', {})
        if constraint_metrics.get('constraint_satisfaction_rate', 1.0) < 0.9:
            recommendations.append("Strengthen constraint validation with additional symbolic rules")
        
        if constraint_metrics.get('average_confidence', 1.0) < 0.7:
            recommendations.append("Improve confidence calibration through additional training")
        
        # Reasoning quality recommendations
        reasoning_metrics = metrics.get('reasoning_metrics', {})
        if reasoning_metrics.get('explanation_coherence', 1.0) < 0.7:
            recommendations.append("Enhance explanation generation with better training data")
        
        # Performance recommendations
        performance_metrics = metrics.get('performance_metrics', {})
        if performance_metrics.get('average_generation_time', 0.0) > 2.0:
            recommendations.append("Optimize model inference speed for production deployment")
        
        # Integration recommendations
        integration_metrics = metrics.get('integration_metrics', {})
        if integration_metrics and integration_metrics.get('compatibility_score', 1.0) < 0.9:
            recommendations.append("Address integration compatibility issues with fine-tuning pipeline")
        
        return recommendations
    
    def _create_error_result(self, test_case: Dict[str, Any], error: str, dataset_name: str, case_idx: int) -> Dict[str, Any]:
        """Create error result for failed test cases"""
        
        return {
            'dataset': dataset_name,
            'case_index': case_idx,
            'instruction': test_case.get('instruction', ''),
            'schema': test_case.get('schema', ''),
            'expected_sql': test_case.get('expected_sql', ''),
            'generated_sql': None,
            'confidence': 0.0,
            'is_valid': False,
            'violations': [f"Evaluation error: {error}"],
            'explanation': None,
            'generation_time': 0.0,
            'sql_correctness_score': 0.0,
            'reasoning_quality_score': 0.0,
            'timestamp': datetime.now().isoformat(),
            'error': error
        }
    
    def _save_results(self, result: EvaluationResult):
        """Save evaluation results to files"""
        
        # Save main result
        result_file = self.output_dir / f"evaluation_result_{result.test_id}.json"
        with open(result_file, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        # Save detailed results as CSV for analysis
        if result.detailed_results:
            csv_file = self.output_dir / f"detailed_results_{result.test_id}.csv"
            df = pd.DataFrame(result.detailed_results)
            df.to_csv(csv_file, index=False)
        
        # Save summary report
        summary_file = self.output_dir / f"summary_report_{result.test_id}.md"
        self._generate_summary_report(result, summary_file)
        
        self.logger.info(f"Results saved: {result_file}")
    
    def _generate_summary_report(self, result: EvaluationResult, output_file: Path):
        """Generate markdown summary report"""
        
        report = f"""# Neurosymbolic SQL Adapter Evaluation Report

## Test Information
- **Test ID**: {result.test_id}
- **Timestamp**: {result.timestamp}
- **Model Type**: {result.model_info.get('model_type', 'Unknown')}
- **Overall Score**: {result.overall_score:.3f}/1.000

## SQL Correctness Metrics
"""
        
        for metric, value in result.sql_correctness.items():
            report += f"- **{metric.replace('_', ' ').title()}**: {value:.3f}\n"
        
        report += f"""
## Constraint Satisfaction Metrics
"""
        
        for metric, value in result.constraint_metrics.items():
            report += f"- **{metric.replace('_', ' ').title()}**: {value:.3f}\n"
        
        report += f"""
## Reasoning Quality Metrics
"""
        
        for metric, value in result.reasoning_metrics.items():
            report += f"- **{metric.replace('_', ' ').title()}**: {value:.3f}\n"
        
        report += f"""
## Performance Metrics
"""
        
        for metric, value in result.performance_metrics.items():
            report += f"- **{metric.replace('_', ' ').title()}**: {value:.3f}\n"
        
        if result.integration_metrics:
            report += f"""
## Integration Metrics
"""
            for metric, value in result.integration_metrics.items():
                report += f"- **{metric.replace('_', ' ').title()}**: {value:.3f}\n"
        
        if result.recommendations:
            report += f"""
## Recommendations
"""
            for i, rec in enumerate(result.recommendations, 1):
                report += f"{i}. {rec}\n"
        
        report += f"""
## Detailed Results
- Total test cases: {len(result.detailed_results) if result.detailed_results else 0}
- Detailed results saved to: `detailed_results_{result.test_id}.csv`

---
*Generated by Neurosymbolic SQL Adapter Evaluation Framework*
"""
        
        with open(output_file, 'w') as f:
            f.write(report)

def create_evaluation_config(
    enable_integration: bool = True,
    fine_tuning_project_path: str = "/Users/saptak/code/fine-tuning-small-llms",
    output_dir: str = "./evaluation_results"
) -> Dict[str, Any]:
    """Create default evaluation configuration"""
    
    return {
        'enable_integration': enable_integration,
        'output_dir': output_dir,
        
        'sql_metrics': {
            'enable_syntax_validation': True,
            'enable_semantic_analysis': True,
            'enable_execution_validation': False  # Set to True if database available
        },
        
        'reasoning_quality': {
            'explanation_methods': ['coherence', 'completeness', 'accuracy'],
            'enable_human_evaluation': False
        },
        
        'performance': {
            'timeout_seconds': 30,
            'memory_tracking': True,
            'detailed_profiling': False
        },
        
        'integration_config': {
            'fine_tuning_project_path': fine_tuning_project_path,
            'model_paths': {
                'sql_expert': 'models/sql-expert/merged_model',
                'base_model': 'models/base-model'
            },
            'enable_comparative_analysis': True,
            'enable_ablation_studies': True
        }
    }