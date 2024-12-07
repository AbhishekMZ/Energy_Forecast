from typing import Dict, Any, List, Optional, Tuple, Callable
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import threading
from queue import Queue
import time
import logging
from datetime import datetime
import traceback
from dataclasses import dataclass, field
from .integration_tests import ModelIntegrationTester, IntegrationTestResult
from ..utils.error_handling import ProcessingError

@dataclass
class ParallelTestResult:
    """Container for parallel test results"""
    test_name: str
    start_time: datetime
    end_time: datetime
    status: bool
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)

class ParallelTestRunner:
    """Run integration tests in parallel with resource management"""
    
    def __init__(self, n_processes: Optional[int] = None,
                 n_threads: Optional[int] = None):
        """
        Initialize parallel test runner
        
        Parameters:
            n_processes: Number of processes (default: CPU count)
            n_threads: Number of threads per process (default: 2)
        """
        self.n_processes = n_processes or multiprocessing.cpu_count()
        self.n_threads = n_threads or 2
        self.logger = logging.getLogger(__name__)
        self.results_queue = multiprocessing.Queue()
        self.resource_monitor = ResourceMonitor()
    
    def _run_test_in_thread(self, test_func: Callable,
                           test_args: Tuple) -> ParallelTestResult:
        """Run a single test in a thread with resource monitoring"""
        start_time = datetime.now()
        
        try:
            # Start resource monitoring
            resource_usage = {}
            self.resource_monitor.start_monitoring()
            
            # Run test
            result = test_func(*test_args)
            
            # Get resource usage
            resource_usage = self.resource_monitor.get_usage()
            
            end_time = datetime.now()
            
            return ParallelTestResult(
                test_name=f"{test_func.__name__}_{test_args}",
                start_time=start_time,
                end_time=end_time,
                status=True if isinstance(result, IntegrationTestResult) else False,
                details=result.details if isinstance(result, IntegrationTestResult) else None,
                resource_usage=resource_usage
            )
            
        except Exception as e:
            end_time = datetime.now()
            return ParallelTestResult(
                test_name=f"{test_func.__name__}_{test_args}",
                start_time=start_time,
                end_time=end_time,
                status=False,
                error_message=str(e),
                resource_usage=self.resource_monitor.get_usage()
            )
        finally:
            self.resource_monitor.stop_monitoring()
    
    def _process_worker(self, test_batch: List[Tuple[Callable, Tuple]]) -> None:
        """Worker function for process pool"""
        thread_pool = ThreadPoolExecutor(max_workers=self.n_threads)
        futures = []
        
        try:
            # Submit tests to thread pool
            for test_func, test_args in test_batch:
                future = thread_pool.submit(
                    self._run_test_in_thread,
                    test_func,
                    test_args
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    self.results_queue.put(result)
                except Exception as e:
                    self.logger.error(f"Error in thread: {str(e)}")
                    
        finally:
            thread_pool.shutdown()
    
    def run_parallel_tests(self,
                          test_cases: List[Tuple[Callable, Tuple]]) -> List[ParallelTestResult]:
        """
        Run tests in parallel using process pool and thread pool
        
        Parameters:
            test_cases: List of (test_function, test_args) tuples
        """
        results = []
        
        # Split test cases into batches for processes
        batch_size = max(1, len(test_cases) // self.n_processes)
        test_batches = [
            test_cases[i:i + batch_size]
            for i in range(0, len(test_cases), batch_size)
        ]
        
        # Create process pool
        with ProcessPoolExecutor(max_workers=self.n_processes) as process_pool:
            # Submit batches to process pool
            futures = [
                process_pool.submit(self._process_worker, batch)
                for batch in test_batches
            ]
            
            # Wait for all processes to complete
            for _ in as_completed(futures):
                # Collect results from queue
                while not self.results_queue.empty():
                    result = self.results_queue.get()
                    results.append(result)
        
        return results
    
    def run_test_suite(self) -> Dict[str, List[ParallelTestResult]]:
        """Run complete test suite in parallel"""
        tester = ModelIntegrationTester()
        
        # Define test cases
        test_cases = [
            (tester.test_data_characteristics, ()),
            (tester.test_auto_config_tuning, ()),
            (tester.test_config_validation, ()),
            (tester.test_visualization, ()),
            (tester.test_end_to_end_pipeline, ())
        ]
        
        # Run tests in parallel
        results = self.run_parallel_tests(test_cases)
        
        # Organize results by category
        categorized_results = {}
        for result in results:
            category = result.test_name.split('_')[0]
            if category not in categorized_results:
                categorized_results[category] = []
            categorized_results[category].append(result)
        
        return categorized_results
    
    def generate_parallel_report(self,
                               results: Dict[str, List[ParallelTestResult]]) -> str:
        """Generate detailed report for parallel test execution"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report_lines = [
            "# Parallel Test Execution Report",
            f"Generated on: {timestamp}\n",
            "## Performance Summary",
            "| Category | Tests | Passed | Failed | Avg Time (s) | Max Memory (MB) |",
            "|----------|-------|---------|---------|--------------|----------------|"
        ]
        
        total_tests = 0
        total_passed = 0
        total_time = 0
        max_memory = 0
        
        for category, category_results in results.items():
            n_tests = len(category_results)
            n_passed = sum(1 for r in category_results if r.status)
            avg_time = np.mean([
                (r.end_time - r.start_time).total_seconds()
                for r in category_results
            ])
            max_mem = max(
                r.resource_usage.get('memory_mb', 0)
                for r in category_results
            )
            
            report_lines.append(
                f"| {category} | {n_tests} | {n_passed} | "
                f"{n_tests - n_passed} | {avg_time:.2f} | {max_mem:.1f} |"
            )
            
            total_tests += n_tests
            total_passed += n_passed
            total_time += avg_time * n_tests
            max_memory = max(max_memory, max_mem)
        
        report_lines.extend([
            "",
            "## Resource Usage Analysis",
            "| Resource | Average | Maximum | Minimum |",
            "|----------|----------|---------|----------|"
        ])
        
        # Aggregate resource usage
        all_results = [r for results in results.values() for r in results]
        for resource in ['cpu_percent', 'memory_mb', 'disk_io_mb']:
            values = [r.resource_usage.get(resource, 0) for r in all_results]
            report_lines.append(
                f"| {resource} | {np.mean(values):.1f} | "
                f"{np.max(values):.1f} | {np.min(values):.1f} |"
            )
        
        report_lines.extend([
            "",
            "## Test Execution Timeline",
            "| Test | Start Time | Duration (s) | Status | Memory (MB) |",
            "|------|------------|--------------|---------|-------------|"
        ])
        
        # Sort all results by start time
        all_results.sort(key=lambda x: x.start_time)
        
        for result in all_results:
            duration = (result.end_time - result.start_time).total_seconds()
            status = "✓" if result.status else "✗"
            memory = result.resource_usage.get('memory_mb', 0)
            
            report_lines.append(
                f"| {result.test_name} | "
                f"{result.start_time.strftime('%H:%M:%S')} | "
                f"{duration:.2f} | {status} | {memory:.1f} |"
            )
        
        report_lines.extend([
            "",
            "## Error Analysis",
            "| Test | Error Message | Stack Trace |",
            "|------|---------------|-------------|"
        ])
        
        for result in all_results:
            if not result.status:
                report_lines.append(
                    f"| {result.test_name} | {result.error_message} | "
                    f"{result.details.get('stack_trace', 'N/A') if result.details else 'N/A'} |"
                )
        
        report_lines.extend([
            "",
            "## Overall Statistics",
            f"- Total Tests: {total_tests}",
            f"- Passed: {total_passed}",
            f"- Failed: {total_tests - total_passed}",
            f"- Total Execution Time: {total_time:.2f}s",
            f"- Maximum Memory Usage: {max_memory:.1f}MB",
            f"- Average Time per Test: {total_time/total_tests:.2f}s",
            f"- Pass Rate: {(total_passed/total_tests*100):.1f}%"
        ])
        
        return "\n".join(report_lines)


class ResourceMonitor:
    """Monitor system resource usage during test execution"""
    
    def __init__(self):
        self.running = False
        self.usage_data = {
            'cpu_percent': [],
            'memory_mb': [],
            'disk_io_mb': []
        }
        self.monitor_thread = None
    
    def _monitor(self) -> None:
        """Monitor resource usage"""
        import psutil
        process = psutil.Process()
        
        while self.running:
            try:
                self.usage_data['cpu_percent'].append(process.cpu_percent())
                self.usage_data['memory_mb'].append(
                    process.memory_info().rss / 1024 / 1024
                )
                disk_io = process.io_counters()
                self.usage_data['disk_io_mb'].append(
                    (disk_io.read_bytes + disk_io.write_bytes) / 1024 / 1024
                )
                time.sleep(0.1)  # Sample every 100ms
                
            except Exception as e:
                self.logger.error(f"Error monitoring resources: {str(e)}")
    
    def start_monitoring(self) -> None:
        """Start resource monitoring"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def get_usage(self) -> Dict[str, float]:
        """Get average resource usage"""
        return {
            metric: np.mean(values) if values else 0
            for metric, values in self.usage_data.items()
        }
