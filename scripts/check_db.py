import os
import sys
import logging
from sqlalchemy import text
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.database import SessionLocal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_connection():
    """Check database connection"""
    db = SessionLocal()
    try:
        # Test connection
        db.execute(text("SELECT 1"))
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False
    finally:
        db.close()

def check_tables():
    """Check if all required tables exist"""
    db = SessionLocal()
    try:
        # Check each table
        tables = ["cities", "consumption_data", "forecasts"]
        for table in tables:
            result = db.execute(
                text(f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{table}')")
            ).scalar()
            if result:
                logger.info(f"Table '{table}' exists")
            else:
                logger.error(f"Table '{table}' does not exist")
                return False
        return True
    except Exception as e:
        logger.error(f"Error checking tables: {e}")
        return False
    finally:
        db.close()

def check_data():
    """Check if tables have data"""
    db = SessionLocal()
    try:
        tables = ["cities", "consumption_data", "forecasts"]
        for table in tables:
            count = db.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            logger.info(f"Table '{table}' has {count} records")
    except Exception as e:
        logger.error(f"Error checking data: {e}")
    finally:
        db.close()

def main():
    """Main health check function"""
    load_dotenv()
    
    # Run checks
    connection_ok = check_connection()
    if connection_ok:
        tables_ok = check_tables()
        if tables_ok:
            check_data()
        
if __name__ == "__main__":
    main()
