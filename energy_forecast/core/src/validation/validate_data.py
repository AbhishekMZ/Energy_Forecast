import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
from basic_validator import BasicDataValidator
from statistical_validator import StatisticalValidator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataValidationPipeline:
    """Complete data validation pipeline combining basic and statistical validation"""
    
    def __init__(self):
        self.basic_validator = BasicDataValidator()
        self.statistical_validator = StatisticalValidator()
    
    def validate_dataset(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """Run complete validation pipeline on dataset"""
        logger.info("Starting data validation pipeline...")
        
        # Run basic validations
        logger.info("Running basic validations...")
        basic_valid, basic_results = self.basic_validator.run_all_validations(df)
        
        if not basic_valid:
            logger.warning("Basic validation failed")
            return False, {'basic': basic_results}
        
        # Run statistical validations
        logger.info("Running statistical validations...")
        stat_valid, stat_results = self.statistical_validator.run_statistical_validations(df)
        
        # Combine results
        all_results = {
            'basic': basic_results,
            'statistical': stat_results
        }
        
        overall_valid = basic_valid and stat_valid
        
        if overall_valid:
            logger.info("All validations passed successfully")
        else:
            logger.warning("Some validations failed")
        
        return overall_valid, all_results
    
    def generate_validation_report(self, validation_results: Dict) -> str:
        """Generate comprehensive validation report"""
        report = []
        report.append("=== Energy Data Validation Report ===\n")
        
        # Basic validation report
        if 'basic' in validation_results:
            report.append("Basic Validation Results:")
            report.append(self.basic_validator.generate_validation_report(
                validation_results['basic']
            ))
        
        # Statistical validation report
        if 'statistical' in validation_results:
            report.append("\nStatistical Validation Results:")
            report.append(self.statistical_validator.generate_statistical_report(
                validation_results['statistical']
            ))
        
        return "\n".join(report)
    
    def save_validation_report(self, report: str, output_path: Path):
        """Save validation report to file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Validation report saved to {output_path}")

def main():
    """Main function to run validation pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate energy forecast data')
    parser.add_argument('input_file', type=str, help='Path to input data file')
    parser.add_argument('--output', type=str, default='validation_report.txt',
                      help='Path to save validation report')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input_file}")
    df = pd.read_csv(args.input_file, parse_dates=['timestamp'])
    
    # Run validation pipeline
    pipeline = DataValidationPipeline()
    valid, results = pipeline.validate_dataset(df)
    
    # Generate and save report
    report = pipeline.generate_validation_report(results)
    pipeline.save_validation_report(report, Path(args.output))
    
    # Exit with appropriate status code
    exit(0 if valid else 1)

if __name__ == '__main__':
    main()
