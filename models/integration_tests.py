import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
from .auto_config import AutoConfigTuner, DataCharacteristics
from .config_validation import ConfigurationValidator, ValidationResult
from .model_configs import ModelConfigurations
from .config_visualization import ConfigurationVisualizer
from ..utils.error_handling import ProcessingError

@dataclass
class IntegrationTestResult:
    """Container for integration test results"""
    test_name: str
    status: bool
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class ModelIntegrationTester:
    """Test integration between different components and handle edge cases"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = []
    
    def _generate_test_data(self, case: str = 'normal') -> pd.DataFrame:
        """Generate test data for different scenarios"""
        np.random.seed(42)
        n_samples = 1000
        
        if case == 'normal':
            # Normal case with clean data
            dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
            data = pd.DataFrame({
                'timestamp': dates,
                'temperature': np.random.normal(20, 5, n_samples),
                'humidity': np.random.uniform(30, 70, n_samples),
                'wind_speed': np.abs(np.random.normal(10, 3, n_samples)),
                'consumption': np.random.normal(100, 20, n_samples)
            })
            
        elif case == 'missing_values':
            # Data with missing values
            data = self._generate_test_data('normal')
            mask = np.random.choice([True, False], size=n_samples, p=[0.1, 0.9])
            data.loc[mask, ['temperature', 'humidity']] = np.nan
            
        elif case == 'outliers':
            # Data with outliers
            data = self._generate_test_data('normal')
            outlier_mask = np.random.choice([True, False], size=n_samples, p=[0.05, 0.95])
            data.loc[outlier_mask, 'consumption'] *= 10
            
        elif case == 'non_stationary':
            # Non-stationary time series
            data = self._generate_test_data('normal')
            trend = np.linspace(0, 50, n_samples)
            data['consumption'] += trend
            
        elif case == 'minimal_data':
            # Very small dataset
            data = self._generate_test_data('normal').head(50)
            
        elif case == 'large_data':
            # Large dataset
            data = pd.concat([self._generate_test_data('normal')] * 10)
            
        elif case == 'high_cardinality':
            # Data with high cardinality features
            data = self._generate_test_data('normal')
            data['category'] = np.random.choice(range(100), size=n_samples)
            
        elif case == 'multicollinear':
            # Data with multicollinear features
            data = self._generate_test_data('normal')
            data['temp_squared'] = data['temperature'] ** 2
            data['humid_temp'] = data['temperature'] * data['humidity']
            
        return data
    
    def test_data_characteristics(self) -> List[IntegrationTestResult]:
        """Test data analysis component"""
        test_cases = ['normal', 'missing_values', 'outliers', 'non_stationary',
                     'minimal_data', 'large_data', 'high_cardinality',
                     'multicollinear']
        
        results = []
        for case in test_cases:
            try:
                data = self._generate_test_data(case)
                analyzer = DataCharacteristics(
                    data=data,
                    target_col='consumption',
                    timestamp_col='timestamp'
                )
                
                # Test basic stats
                basic_stats = analyzer.analyze_basic_stats()
                assert isinstance(basic_stats, dict)
                
                # Test time series analysis
                ts_chars = analyzer.analyze_time_series_characteristics()
                if case != 'minimal_data':
                    assert ts_chars is not None
                
                # Test feature relationships
                relationships = analyzer.analyze_feature_relationships()
                assert isinstance(relationships, dict)
                
                results.append(IntegrationTestResult(
                    test_name=f"data_characteristics_{case}",
                    status=True,
                    details={'stats': basic_stats}
                ))
                
            except Exception as e:
                results.append(IntegrationTestResult(
                    test_name=f"data_characteristics_{case}",
                    status=False,
                    error_message=str(e)
                ))
        
        return results
    
    def test_auto_config_tuning(self) -> List[IntegrationTestResult]:
        """Test automatic configuration tuning"""
        test_cases = ['normal', 'minimal_data', 'large_data', 'high_cardinality']
        
        results = []
        for case in test_cases:
            try:
                data = self._generate_test_data(case)
                tuner = AutoConfigTuner(
                    data=data,
                    target_col='consumption',
                    timestamp_col='timestamp'
                )
                
                # Test data analysis
                tuner.analyze_data()
                assert tuner.complexity_score is not None
                
                # Test configuration generation
                configs = tuner.get_optimal_model_configs()
                assert isinstance(configs, dict)
                assert len(configs) > 0
                
                # Verify configuration adaptation
                if case == 'minimal_data':
                    assert configs['random_forest']['n_estimators'] <= 100
                elif case == 'large_data':
                    assert configs['random_forest']['n_estimators'] >= 200
                
                results.append(IntegrationTestResult(
                    test_name=f"auto_config_{case}",
                    status=True,
                    details={'configs': configs}
                ))
                
            except Exception as e:
                results.append(IntegrationTestResult(
                    test_name=f"auto_config_{case}",
                    status=False,
                    error_message=str(e)
                ))
        
        return results
    
    def test_config_validation(self) -> List[IntegrationTestResult]:
        """Test configuration validation"""
        test_cases = ['normal', 'missing_values', 'outliers']
        
        results = []
        for case in test_cases:
            try:
                data = self._generate_test_data(case)
                validator = ConfigurationValidator(
                    data=data,
                    target_col='consumption',
                    timestamp_col='timestamp'
                )
                
                # Test validation with different fold counts
                for n_splits in [3, 5]:
                    validation_results = validator.validate_configurations(
                        n_splits=n_splits,
                        n_trials=2  # Reduced for testing
                    )
                    
                    assert isinstance(validation_results, list)
                    assert len(validation_results) > 0
                    assert all(isinstance(r, ValidationResult)
                             for r in validation_results)
                
                results.append(IntegrationTestResult(
                    test_name=f"config_validation_{case}",
                    status=True,
                    details={'n_results': len(validation_results)}
                ))
                
            except Exception as e:
                results.append(IntegrationTestResult(
                    test_name=f"config_validation_{case}",
                    status=False,
                    error_message=str(e)
                ))
        
        return results
    
    def test_visualization(self) -> List[IntegrationTestResult]:
        """Test visualization capabilities"""
        test_cases = ['normal']  # Limited cases for visualization testing
        
        results = []
        for case in test_cases:
            try:
                data = self._generate_test_data(case)
                
                # Get validation results
                validator = ConfigurationValidator(
                    data=data,
                    target_col='consumption',
                    timestamp_col='timestamp'
                )
                validation_results = validator.validate_configurations(
                    n_splits=3,
                    n_trials=2
                )
                
                # Test visualization
                visualizer = ConfigurationVisualizer()
                
                # Test each plot type
                plots = [
                    visualizer.plot_performance_comparison(validation_results),
                    visualizer.plot_parameter_importance(validation_results),
                    visualizer.plot_validation_stability(validation_results),
                    visualizer.plot_configuration_evolution(validation_results)
                ]
                
                assert all(plot is not None for plot in plots)
                
                # Test dashboard creation
                dashboard = visualizer.create_dashboard(validation_results)
                assert dashboard is not None
                
                results.append(IntegrationTestResult(
                    test_name=f"visualization_{case}",
                    status=True,
                    details={'n_plots': len(plots)}
                ))
                
            except Exception as e:
                results.append(IntegrationTestResult(
                    test_name=f"visualization_{case}",
                    status=False,
                    error_message=str(e)
                ))
        
        return results
    
    def test_end_to_end_pipeline(self) -> IntegrationTestResult:
        """Test complete end-to-end pipeline"""
        try:
            # Generate test data
            data = self._generate_test_data('normal')
            
            # 1. Data Analysis
            analyzer = DataCharacteristics(
                data=data,
                target_col='consumption',
                timestamp_col='timestamp'
            )
            characteristics = analyzer.analyze_basic_stats()
            ts_chars = analyzer.analyze_time_series_characteristics()
            
            # 2. Configuration Tuning
            tuner = AutoConfigTuner(
                data=data,
                target_col='consumption',
                timestamp_col='timestamp'
            )
            tuner.analyze_data()
            configs = tuner.get_optimal_model_configs()
            
            # 3. Configuration Validation
            validator = ConfigurationValidator(
                data=data,
                target_col='consumption',
                timestamp_col='timestamp'
            )
            validation_results = validator.validate_configurations(
                n_splits=3,
                n_trials=2
            )
            
            # 4. Visualization
            visualizer = ConfigurationVisualizer()
            dashboard = visualizer.create_dashboard(validation_results)
            
            return IntegrationTestResult(
                test_name="end_to_end_pipeline",
                status=True,
                details={
                    'data_size': len(data),
                    'n_configs': len(configs),
                    'n_validation_results': len(validation_results)
                }
            )
            
        except Exception as e:
            return IntegrationTestResult(
                test_name="end_to_end_pipeline",
                status=False,
                error_message=str(e)
            )
    
    def run_all_tests(self) -> Dict[str, List[IntegrationTestResult]]:
        """Run all integration tests"""
        all_results = {
            'data_characteristics': self.test_data_characteristics(),
            'auto_config': self.test_auto_config_tuning(),
            'config_validation': self.test_config_validation(),
            'visualization': self.test_visualization(),
            'end_to_end': [self.test_end_to_end_pipeline()]
        }
        
        # Log results
        for category, results in all_results.items():
            self.logger.info(f"\nResults for {category}:")
            for result in results:
                status = "✓" if result.status else "✗"
                self.logger.info(f"{status} {result.test_name}")
                if not result.status:
                    self.logger.error(f"  Error: {result.error_message}")
        
        return all_results
    
    def generate_test_report(self,
                           results: Dict[str, List[IntegrationTestResult]]) -> str:
        """Generate markdown report of test results"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report_lines = [
            "# Model Integration Test Report",
            f"Generated on: {timestamp}\n",
            "## Summary",
            "| Category | Tests | Passed | Failed |",
            "|----------|-------|---------|---------|"
        ]
        
        total_tests = 0
        total_passed = 0
        
        for category, category_results in results.items():
            n_tests = len(category_results)
            n_passed = sum(1 for r in category_results if r.status)
            n_failed = n_tests - n_passed
            
            report_lines.append(
                f"| {category} | {n_tests} | {n_passed} | {n_failed} |"
            )
            
            total_tests += n_tests
            total_passed += n_passed
        
        report_lines.extend([
            "",
            "## Detailed Results\n"
        ])
        
        for category, category_results in results.items():
            report_lines.extend([
                f"### {category.replace('_', ' ').title()}",
                "| Test | Status | Details |",
                "|------|--------|----------|"
            ])
            
            for result in category_results:
                status = "✓" if result.status else "✗"
                details = (f"Error: {result.error_message}"
                          if not result.status
                          else str(result.details or ""))
                report_lines.append(
                    f"| {result.test_name} | {status} | {details} |"
                )
            
            report_lines.append("")
        
        report_lines.extend([
            "## Overall Statistics",
            f"- Total Tests: {total_tests}",
            f"- Passed: {total_passed}",
            f"- Failed: {total_tests - total_passed}",
            f"- Pass Rate: {(total_passed / total_tests * 100):.1f}%"
        ])
        
        return "\n".join(report_lines)
    
    def save_test_report(self, report: str, filename: str) -> None:
        """Save test report to file"""
        with open(filename, 'w') as f:
            f.write(report)
