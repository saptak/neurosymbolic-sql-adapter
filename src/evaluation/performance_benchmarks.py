#!/usr/bin/env python3
"""
Performance Benchmarking Module

This module provides comprehensive performance evaluation for neurosymbolic SQL adapters,
including speed, memory usage, throughput, and comparative analysis capabilities.
"""

import time
import psutil
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics
import concurrent.futures

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@dataclass
class PerformanceResult:
    """Result container for performance benchmarking"""
    
    # Timing metrics
    average_inference_time: float
    median_inference_time: float
    min_inference_time: float
    max_inference_time: float
    p95_inference_time: float
    
    # Throughput metrics
    queries_per_second: float
    tokens_per_second: float
    
    # Memory metrics
    peak_memory_usage_mb: float
    average_memory_usage_mb: float
    memory_efficiency_score: float
    
    # GPU metrics (if available)
    gpu_utilization: Optional[float] = None
    gpu_memory_usage_mb: Optional[float] = None
    
    # System metrics
    cpu_utilization: float = 0.0
    
    # Quality vs Performance trade-off
    performance_quality_score: float = 0.0
    
    # Detailed measurements
    individual_timings: List[float] = None
    memory_timeline: List[float] = None

class PerformanceBenchmarks:
    """
    Comprehensive performance benchmarking system
    
    Provides multiple benchmarking capabilities:
    1. Inference speed measurement
    2. Memory usage tracking
    3. Throughput analysis
    4. Comparative performance analysis
    5. Scalability testing
    6. Resource utilization monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize performance benchmarking system
        
        Args:
            config: Configuration including:
                - timeout_seconds: Maximum time for performance tests
                - memory_tracking: Whether to track memory usage
                - detailed_profiling: Whether to enable detailed profiling
                - gpu_monitoring: Whether to monitor GPU usage
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Configuration
        self.timeout_seconds = config.get('timeout_seconds', 30)
        self.memory_tracking = config.get('memory_tracking', True)
        self.detailed_profiling = config.get('detailed_profiling', False)
        self.gpu_monitoring = config.get('gpu_monitoring', TORCH_AVAILABLE)
        
        # Performance tracking
        self.performance_history = []
        self.current_benchmark = None
        
        # System monitoring
        self.system_monitor = SystemMonitor() if self.memory_tracking else None
        
        self.logger.info("Performance benchmarking system initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for performance benchmarks"""
        logger = logging.getLogger("performance_benchmarks")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def benchmark_model(
        self,
        model: Any,
        test_cases: List[Dict[str, Any]],
        warmup_runs: int = 5,
        benchmark_runs: int = 50
    ) -> Dict[str, Any]:
        """
        Comprehensive model performance benchmark
        
        Args:
            model: Model to benchmark
            test_cases: List of test cases for benchmarking
            warmup_runs: Number of warmup runs
            benchmark_runs: Number of benchmark runs
            
        Returns:
            Dictionary containing performance metrics
        """
        
        self.logger.info(f"Starting performance benchmark: {warmup_runs} warmup + {benchmark_runs} benchmark runs")
        
        if not test_cases:
            self.logger.error("No test cases provided for benchmarking")
            return self._create_empty_result("No test cases")
        
        # Start system monitoring
        if self.system_monitor:
            self.system_monitor.start_monitoring()
        
        try:
            # Warmup phase
            self.logger.info("Running warmup phase...")
            self._run_warmup(model, test_cases[:min(len(test_cases), warmup_runs)])
            
            # Benchmark phase
            self.logger.info("Running benchmark phase...")
            benchmark_results = self._run_benchmark(model, test_cases, benchmark_runs)
            
            # Calculate comprehensive metrics
            performance_result = self._calculate_performance_metrics(benchmark_results)
            
            return self._result_to_dict(performance_result)
            
        except Exception as e:
            self.logger.error(f"Performance benchmark failed: {e}")
            return self._create_error_result(str(e))
            
        finally:
            # Stop system monitoring
            if self.system_monitor:
                self.system_monitor.stop_monitoring()
    
    def _run_warmup(self, model: Any, warmup_cases: List[Dict[str, Any]]):
        """Run warmup phase to stabilize performance"""
        
        for i, test_case in enumerate(warmup_cases):
            try:
                instruction = test_case.get('instruction', '')
                schema = test_case.get('schema', '')
                
                # Run inference without timing
                if hasattr(model, 'generate_sql'):
                    model.generate_sql(instruction, schema)
                else:
                    # Mock inference
                    time.sleep(0.01)
                
                if i % 2 == 0:
                    self.logger.debug(f"Warmup progress: {i+1}/{len(warmup_cases)}")
                    
            except Exception as e:
                self.logger.warning(f"Warmup run {i} failed: {e}")
    
    def _run_benchmark(
        self, 
        model: Any, 
        test_cases: List[Dict[str, Any]], 
        benchmark_runs: int
    ) -> List[Dict[str, Any]]:
        """Run benchmark measurements"""
        
        benchmark_results = []
        
        # Prepare test cases (cycle through if we need more runs)
        extended_cases = (test_cases * ((benchmark_runs // len(test_cases)) + 1))[:benchmark_runs]
        
        for i, test_case in enumerate(extended_cases):
            try:
                # Single benchmark run
                run_result = self._benchmark_single_run(model, test_case, i)
                benchmark_results.append(run_result)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Benchmark progress: {i+1}/{benchmark_runs}")
                    
            except Exception as e:
                self.logger.warning(f"Benchmark run {i} failed: {e}")
                # Add error result
                benchmark_results.append({
                    'run_id': i,
                    'inference_time': float('inf'),
                    'success': False,
                    'error': str(e),
                    'memory_usage': 0.0,
                    'output_length': 0
                })
        
        return benchmark_results
    
    def _benchmark_single_run(self, model: Any, test_case: Dict[str, Any], run_id: int) -> Dict[str, Any]:
        """Benchmark a single inference run"""
        
        instruction = test_case.get('instruction', '')
        schema = test_case.get('schema', '')
        
        # Memory before inference
        memory_before = self._get_current_memory_usage()
        gpu_memory_before = self._get_gpu_memory_usage() if self.gpu_monitoring else None
        
        # Time the inference
        start_time = time.perf_counter()
        
        try:
            # Run inference
            if hasattr(model, 'generate_sql'):
                result = model.generate_sql(instruction, schema)
                output = result.sql if hasattr(result, 'sql') else result.get('sql', '')
            else:
                # Mock inference with realistic timing
                time.sleep(0.05 + len(instruction) * 0.001)
                output = f"SELECT * FROM table WHERE condition = '{instruction[:20]}';"
            
            inference_time = time.perf_counter() - start_time
            success = True
            
        except Exception as e:
            inference_time = time.perf_counter() - start_time
            output = ""
            success = False
            error = str(e)
        
        # Memory after inference
        memory_after = self._get_current_memory_usage()
        gpu_memory_after = self._get_gpu_memory_usage() if self.gpu_monitoring else None
        
        # Calculate memory delta
        memory_delta = memory_after - memory_before if memory_after and memory_before else 0.0
        gpu_memory_delta = (gpu_memory_after - gpu_memory_before) if (gpu_memory_after and gpu_memory_before) else None
        
        return {
            'run_id': run_id,
            'inference_time': inference_time,
            'success': success,
            'memory_usage': memory_after,
            'memory_delta': memory_delta,
            'gpu_memory_usage': gpu_memory_after,
            'gpu_memory_delta': gpu_memory_delta,
            'output_length': len(output) if output else 0,
            'input_length': len(instruction),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_performance_metrics(self, benchmark_results: List[Dict[str, Any]]) -> PerformanceResult:
        """Calculate comprehensive performance metrics from benchmark results"""
        
        # Filter successful runs
        successful_runs = [r for r in benchmark_results if r.get('success', False)]
        
        if not successful_runs:
            self.logger.error("No successful benchmark runs")
            return self._create_empty_performance_result()
        
        # Extract timing data
        inference_times = [r['inference_time'] for r in successful_runs]
        memory_usages = [r['memory_usage'] for r in successful_runs if r['memory_usage']]
        output_lengths = [r['output_length'] for r in successful_runs]
        
        # Calculate timing statistics
        avg_inference_time = statistics.mean(inference_times)
        median_inference_time = statistics.median(inference_times)
        min_inference_time = min(inference_times)
        max_inference_time = max(inference_times)
        
        # Calculate percentiles
        inference_times_sorted = sorted(inference_times)
        p95_index = int(0.95 * len(inference_times_sorted))
        p95_inference_time = inference_times_sorted[p95_index] if inference_times_sorted else 0.0
        
        # Calculate throughput
        total_time = sum(inference_times)
        queries_per_second = len(successful_runs) / total_time if total_time > 0 else 0.0
        
        # Calculate tokens per second (estimate)
        total_output_tokens = sum(output_lengths)
        tokens_per_second = total_output_tokens / total_time if total_time > 0 else 0.0
        
        # Calculate memory statistics
        if memory_usages:
            peak_memory_usage_mb = max(memory_usages)
            average_memory_usage_mb = statistics.mean(memory_usages)
        else:
            peak_memory_usage_mb = 0.0
            average_memory_usage_mb = 0.0
        
        # Calculate memory efficiency (output per MB)
        memory_efficiency_score = (
            total_output_tokens / peak_memory_usage_mb 
            if peak_memory_usage_mb > 0 else 0.0
        )
        
        # GPU metrics
        gpu_utilization = None
        gpu_memory_usage_mb = None
        
        if self.gpu_monitoring:
            gpu_memories = [r['gpu_memory_usage'] for r in successful_runs if r.get('gpu_memory_usage')]
            if gpu_memories:
                gpu_memory_usage_mb = max(gpu_memories)
                gpu_utilization = self._estimate_gpu_utilization()
        
        # CPU utilization
        cpu_utilization = self._get_cpu_utilization()
        
        # Performance-quality trade-off score
        performance_quality_score = self._calculate_performance_quality_score(
            avg_inference_time, peak_memory_usage_mb, len(successful_runs) / len(benchmark_results)
        )
        
        return PerformanceResult(
            average_inference_time=avg_inference_time,
            median_inference_time=median_inference_time,
            min_inference_time=min_inference_time,
            max_inference_time=max_inference_time,
            p95_inference_time=p95_inference_time,
            
            queries_per_second=queries_per_second,
            tokens_per_second=tokens_per_second,
            
            peak_memory_usage_mb=peak_memory_usage_mb,
            average_memory_usage_mb=average_memory_usage_mb,
            memory_efficiency_score=memory_efficiency_score,
            
            gpu_utilization=gpu_utilization,
            gpu_memory_usage_mb=gpu_memory_usage_mb,
            
            cpu_utilization=cpu_utilization,
            performance_quality_score=performance_quality_score,
            
            individual_timings=inference_times,
            memory_timeline=memory_usages
        )
    
    def benchmark_scalability(
        self,
        model: Any,
        test_cases: List[Dict[str, Any]],
        concurrent_levels: List[int] = [1, 2, 4, 8]
    ) -> Dict[str, Any]:
        """
        Benchmark model scalability with concurrent requests
        
        Args:
            model: Model to benchmark
            test_cases: Test cases for benchmarking
            concurrent_levels: List of concurrency levels to test
            
        Returns:
            Dictionary containing scalability metrics
        """
        
        self.logger.info(f"Starting scalability benchmark with concurrency levels: {concurrent_levels}")
        
        scalability_results = {}
        
        for concurrent_level in concurrent_levels:
            self.logger.info(f"Testing concurrency level: {concurrent_level}")
            
            try:
                # Run concurrent benchmark
                concurrent_result = self._benchmark_concurrent_requests(
                    model, test_cases, concurrent_level
                )
                
                scalability_results[f"concurrent_{concurrent_level}"] = concurrent_result
                
            except Exception as e:
                self.logger.error(f"Concurrency level {concurrent_level} failed: {e}")
                scalability_results[f"concurrent_{concurrent_level}"] = {
                    'error': str(e),
                    'success': False
                }
        
        # Calculate scalability metrics
        scalability_metrics = self._calculate_scalability_metrics(scalability_results)
        
        return {
            'scalability_results': scalability_results,
            'scalability_metrics': scalability_metrics
        }
    
    def _benchmark_concurrent_requests(
        self,
        model: Any,
        test_cases: List[Dict[str, Any]],
        concurrent_level: int,
        requests_per_thread: int = 10
    ) -> Dict[str, Any]:
        """Benchmark concurrent requests at specific concurrency level"""
        
        def worker_thread(thread_id: int, test_cases_subset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Worker thread for concurrent benchmarking"""
            thread_results = []
            
            for i, test_case in enumerate(test_cases_subset):
                try:
                    instruction = test_case.get('instruction', '')
                    schema = test_case.get('schema', '')
                    
                    start_time = time.perf_counter()
                    
                    if hasattr(model, 'generate_sql'):
                        result = model.generate_sql(instruction, schema)
                        output = result.sql if hasattr(result, 'sql') else result.get('sql', '')
                    else:
                        time.sleep(0.05)  # Mock processing time
                        output = f"SELECT * FROM table{thread_id};"
                    
                    inference_time = time.perf_counter() - start_time
                    
                    thread_results.append({
                        'thread_id': thread_id,
                        'request_id': i,
                        'inference_time': inference_time,
                        'success': True,
                        'output_length': len(output) if output else 0
                    })
                    
                except Exception as e:
                    thread_results.append({
                        'thread_id': thread_id,
                        'request_id': i,
                        'inference_time': 0.0,
                        'success': False,
                        'error': str(e),
                        'output_length': 0
                    })
            
            return thread_results
        
        # Prepare test cases for each thread
        test_cases_per_thread = test_cases[:requests_per_thread]
        
        # Start concurrent execution
        start_time = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_level) as executor:
            # Submit jobs
            futures = []
            for thread_id in range(concurrent_level):
                future = executor.submit(worker_thread, thread_id, test_cases_per_thread)
                futures.append(future)
            
            # Collect results
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    thread_results = future.result()
                    all_results.extend(thread_results)
                except Exception as e:
                    self.logger.error(f"Thread execution failed: {e}")
        
        total_time = time.perf_counter() - start_time
        
        # Calculate concurrent metrics
        successful_requests = [r for r in all_results if r.get('success', False)]
        
        if successful_requests:
            avg_response_time = statistics.mean([r['inference_time'] for r in successful_requests])
            total_throughput = len(successful_requests) / total_time
            success_rate = len(successful_requests) / len(all_results)
        else:
            avg_response_time = 0.0
            total_throughput = 0.0
            success_rate = 0.0
        
        return {
            'concurrent_level': concurrent_level,
            'total_requests': len(all_results),
            'successful_requests': len(successful_requests),
            'success_rate': success_rate,
            'total_time': total_time,
            'average_response_time': avg_response_time,
            'total_throughput': total_throughput,
            'requests_per_second': total_throughput,
            'detailed_results': all_results
        }
    
    def _calculate_scalability_metrics(self, scalability_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate scalability metrics from concurrent benchmark results"""
        
        metrics = {}
        
        # Extract throughput for each concurrency level
        throughputs = {}
        response_times = {}
        
        for level_key, result in scalability_results.items():
            if result.get('success', True) and 'concurrent_' in level_key:
                concurrent_level = result.get('concurrent_level', 1)
                throughput = result.get('total_throughput', 0.0)
                response_time = result.get('average_response_time', 0.0)
                
                throughputs[concurrent_level] = throughput
                response_times[concurrent_level] = response_time
        
        if throughputs:
            # Calculate scalability efficiency
            baseline_throughput = throughputs.get(1, 0.0)
            if baseline_throughput > 0:
                scalability_factors = {}
                for level, throughput in throughputs.items():
                    scalability_factors[level] = throughput / baseline_throughput
                
                metrics['max_scalability_factor'] = max(scalability_factors.values())
                metrics['avg_scalability_factor'] = statistics.mean(scalability_factors.values())
            
            # Peak throughput
            metrics['peak_throughput'] = max(throughputs.values())
            metrics['optimal_concurrency_level'] = max(throughputs.keys(), key=lambda k: throughputs[k])
            
            # Response time analysis
            if response_times:
                metrics['min_response_time'] = min(response_times.values())  
                metrics['max_response_time'] = max(response_times.values())
                metrics['response_time_variance'] = statistics.variance(response_times.values()) if len(response_times) > 1 else 0.0
        
        return metrics
    
    def compare_models(
        self,
        models: Dict[str, Any],
        test_cases: List[Dict[str, Any]],
        benchmark_runs: int = 30
    ) -> Dict[str, Any]:
        """
        Compare performance of multiple models
        
        Args:
            models: Dictionary of models to compare {name: model}
            test_cases: Test cases for comparison
            benchmark_runs: Number of runs per model
            
        Returns:
            Dictionary containing comparative analysis
        """
        
        self.logger.info(f"Starting comparative performance analysis of {len(models)} models")
        
        model_results = {}
        
        for model_name, model in models.items():
            self.logger.info(f"Benchmarking model: {model_name}")
            
            try:
                model_performance = self.benchmark_model(model, test_cases, warmup_runs=5, benchmark_runs=benchmark_runs)
                model_results[model_name] = model_performance
                
            except Exception as e:
                self.logger.error(f"Failed to benchmark {model_name}: {e}")
                model_results[model_name] = self._create_error_result(str(e))
        
        # Generate comparative analysis
        comparative_analysis = self._generate_comparative_analysis(model_results)
        
        return {
            'individual_results': model_results,
            'comparative_analysis': comparative_analysis
        }
    
    def _generate_comparative_analysis(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative analysis of model performance"""
        
        analysis = {}
        
        # Extract key metrics for comparison
        metrics_to_compare = [
            'average_inference_time',
            'queries_per_second',
            'peak_memory_usage_mb',
            'performance_quality_score'
        ]
        
        for metric in metrics_to_compare:
            metric_values = {}
            
            for model_name, results in model_results.items():
                if metric in results and isinstance(results[metric], (int, float)):
                    metric_values[model_name] = results[metric]
            
            if metric_values:
                # Find best and worst
                if 'time' in metric or 'memory' in metric:
                    # Lower is better
                    best_model = min(metric_values.keys(), key=lambda k: metric_values[k])
                    worst_model = max(metric_values.keys(), key=lambda k: metric_values[k])
                else:
                    # Higher is better
                    best_model = max(metric_values.keys(), key=lambda k: metric_values[k])
                    worst_model = min(metric_values.keys(), key=lambda k: metric_values[k])
                
                analysis[f'{metric}_comparison'] = {
                    'best_model': best_model,
                    'best_value': metric_values[best_model],
                    'worst_model': worst_model,
                    'worst_value': metric_values[worst_model],
                    'all_values': metric_values
                }
        
        # Overall performance ranking
        if model_results:
            performance_scores = {}
            for model_name, results in model_results.items():
                score = results.get('performance_quality_score', 0.0)
                performance_scores[model_name] = score
            
            # Rank models by performance score
            ranked_models = sorted(performance_scores.keys(), key=lambda k: performance_scores[k], reverse=True)
            
            analysis['overall_ranking'] = {
                'ranked_models': ranked_models,
                'performance_scores': performance_scores
            }
        
        return analysis
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0
    
    def _get_gpu_memory_usage(self) -> Optional[float]:
        """Get current GPU memory usage in MB"""
        
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return None
        
        try:
            return torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
        except Exception:
            return None
    
    def _estimate_gpu_utilization(self) -> Optional[float]:
        """Estimate GPU utilization (simplified)"""
        
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return None
        
        # This is a simplified estimation
        # In practice, would use nvidia-ml-py or similar
        try:
            # Check if GPU memory is being used
            memory_used = torch.cuda.memory_allocated()
            memory_total = torch.cuda.get_device_properties(0).total_memory
            
            utilization = (memory_used / memory_total) if memory_total > 0 else 0.0
            return min(1.0, utilization)
            
        except Exception:
            return None
    
    def _get_cpu_utilization(self) -> float:
        """Get current CPU utilization"""
        
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0
    
    def _calculate_performance_quality_score(
        self,
        avg_inference_time: float,
        peak_memory_usage: float,
        success_rate: float
    ) -> float:
        """Calculate combined performance-quality score"""
        
        # Normalize metrics (lower is better for time and memory)
        time_score = max(0.0, 1.0 - min(avg_inference_time / 5.0, 1.0))  # 5s = 0 score
        memory_score = max(0.0, 1.0 - min(peak_memory_usage / 8192, 1.0))  # 8GB = 0 score
        
        # Combined score
        performance_score = (time_score * 0.4 + memory_score * 0.3 + success_rate * 0.3)
        
        return performance_score
    
    def _create_empty_performance_result(self) -> PerformanceResult:
        """Create empty performance result"""
        
        return PerformanceResult(
            average_inference_time=0.0,
            median_inference_time=0.0,
            min_inference_time=0.0,
            max_inference_time=0.0,
            p95_inference_time=0.0,
            
            queries_per_second=0.0,
            tokens_per_second=0.0,
            
            peak_memory_usage_mb=0.0,
            average_memory_usage_mb=0.0,
            memory_efficiency_score=0.0,
            
            cpu_utilization=0.0,
            performance_quality_score=0.0,
            
            individual_timings=[],
            memory_timeline=[]
        )
    
    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """Create empty result with reason"""
        
        result = self._result_to_dict(self._create_empty_performance_result())
        result['error'] = reason
        result['success'] = False
        
        return result
    
    def _create_error_result(self, error: str) -> Dict[str, Any]:
        """Create error result"""
        
        return {
            'error': error,
            'success': False,
            'average_inference_time': float('inf'),
            'queries_per_second': 0.0,
            'peak_memory_usage_mb': 0.0,
            'performance_quality_score': 0.0
        }
    
    def _result_to_dict(self, result: PerformanceResult) -> Dict[str, Any]:
        """Convert performance result to dictionary"""
        
        return {
            'average_inference_time': result.average_inference_time,
            'median_inference_time': result.median_inference_time,
            'min_inference_time': result.min_inference_time,
            'max_inference_time': result.max_inference_time,
            'p95_inference_time': result.p95_inference_time,
            
            'queries_per_second': result.queries_per_second,
            'tokens_per_second': result.tokens_per_second,
            
            'peak_memory_usage_mb': result.peak_memory_usage_mb,
            'average_memory_usage_mb': result.average_memory_usage_mb,
            'memory_efficiency_score': result.memory_efficiency_score,
            
            'gpu_utilization': result.gpu_utilization,
            'gpu_memory_usage_mb': result.gpu_memory_usage_mb,
            
            'cpu_utilization': result.cpu_utilization,
            'performance_quality_score': result.performance_quality_score,
            
            'success': True
        }
    
    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate performance results from multiple benchmarks"""
        
        if not results:
            return {}
        
        # Filter successful results
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {'error': 'No successful benchmark results'}
        
        # Calculate aggregates
        metrics = [
            'average_inference_time', 'queries_per_second', 'peak_memory_usage_mb',
            'memory_efficiency_score', 'performance_quality_score'
        ]
        
        aggregated = {}
        
        for metric in metrics:
            values = [r.get(metric, 0.0) for r in successful_results if isinstance(r.get(metric), (int, float))]
            if values:
                aggregated[f'avg_{metric}'] = statistics.mean(values)
                aggregated[f'min_{metric}'] = min(values)
                aggregated[f'max_{metric}'] = max(values)
                
                if len(values) > 1:
                    aggregated[f'std_{metric}'] = statistics.stdev(values)
        
        # Additional aggregates
        aggregated['total_benchmarks'] = len(successful_results)
        aggregated['success_rate'] = len(successful_results) / len(results)
        
        return aggregated


class SystemMonitor:
    """System resource monitoring utility"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = []
    
    def start_monitoring(self, interval: float = 0.5):
        """Start system monitoring"""
        
        if self.monitoring:
            return
        
        self.monitoring = True
        self.metrics_history = []
        
        def monitor_loop():
            while self.monitoring:
                try:
                    metrics = {
                        'timestamp': time.time(),
                        'cpu_percent': psutil.cpu_percent(),
                        'memory_mb': psutil.virtual_memory().used / 1024 / 1024,
                        'memory_percent': psutil.virtual_memory().percent
                    }
                    
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        metrics['gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
                    
                    self.metrics_history.append(metrics)
                    
                except Exception:
                    pass  # Continue monitoring even if individual readings fail
                
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def get_peak_memory_usage(self) -> float:
        """Get peak memory usage during monitoring"""
        
        if not self.metrics_history:
            return 0.0
        
        return max(m['memory_mb'] for m in self.metrics_history)
    
    def get_average_cpu_usage(self) -> float:
        """Get average CPU usage during monitoring"""
        
        if not self.metrics_history:
            return 0.0
        
        return statistics.mean(m['cpu_percent'] for m in self.metrics_history)