"""Main script to run the energy forecasting pipeline."""

import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

from models.integrated_pipeline import IntegratedPipeline

def setup_logging(config):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format'],
        filename=config['logging']['file']
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run energy forecasting pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to input data file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='ensemble',
        choices=['random_forest', 'lightgbm', 'xgboost', 
                'deep_learning', 'ensemble'],
        help='Model type to use'
    )
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Optimize model configuration'
    )
    return parser.parse_args()

def main():
    """Main function to run pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize pipeline
        logger.info("Initializing pipeline...")
        pipeline = IntegratedPipeline(args.config)
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(args.output) / f'experiment_{timestamp}'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run pipeline
        logger.info(f"Running pipeline with {args.model} model...")
        results = pipeline.run_pipeline(
            data_path=args.data,
            target_col=config['data']['target_column'],
            model_type=args.model,
            optimize_config=args.optimize
        )
        
        # Save results
        logger.info("Saving results...")
        pipeline.save_results(results, str(output_dir))
        
        logger.info(f"Pipeline completed successfully. Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
