import os
import sys
from datetime import datetime
import logging

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_generator import RealisticIndianEnergyGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Create data directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Generate 5 years of data (1 year historical + 4 years for training)
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    logger.info("Initializing data generator...")
    generator = RealisticIndianEnergyGenerator(start_date, end_date)
    
    logger.info("Generating dataset...")
    df = generator.generate_dataset()
    
    # Save raw data
    raw_file = 'data/raw/indian_energy_data_raw.csv'
    df.to_csv(raw_file, index=False)
    logger.info(f"Raw data saved to {raw_file}")
    
    # Print dataset information
    logger.info(f"\nDataset shape: {df.shape}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Cities included: {df['city'].unique().tolist()}")
    logger.info("\nFeature summary:")
    
    # Calculate and print some basic statistics
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        stats = df[col].describe()
        logger.info(f"\n{col}:")
        logger.info(f"  Mean: {stats['mean']:.2f}")
        logger.info(f"  Std: {stats['std']:.2f}")
        logger.info(f"  Min: {stats['min']:.2f}")
        logger.info(f"  Max: {stats['max']:.2f}")
    
    # Calculate city-wise statistics
    logger.info("\nCity-wise average daily demand:")
    for city in df['city'].unique():
        city_data = df[df['city'] == city]
        avg_demand = city_data['total_demand'].mean()
        peak_demand = city_data['peak_demand'].max()
        logger.info(f"{city}:")
        logger.info(f"  Average demand: {avg_demand:.2f} MW")
        logger.info(f"  Peak demand: {peak_demand:.2f} MW")

if __name__ == "__main__":
    main()
