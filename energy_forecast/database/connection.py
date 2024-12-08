"""Database connection management with connection pooling."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import os
from typing import Generator
import logging

logger = logging.getLogger(__name__)

class DatabaseConnection:
    def __init__(self):
        self.engine = create_engine(
            os.getenv('DATABASE_URL'),
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600,
            echo=False
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

    @contextmanager
    def get_session(self) -> Generator:
        """Get a database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            session.close()

    def check_connection(self) -> bool:
        """Check if database connection is alive."""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Connection check failed: {str(e)}")
            return False

    def get_pool_status(self) -> dict:
        """Get current connection pool status."""
        return {
            "pool_size": self.engine.pool.size(),
            "checkedin": self.engine.pool.checkedin(),
            "checkedout": self.engine.pool.checkedout(),
            "overflow": self.engine.pool.overflow()
        }

# Global database instance
db = DatabaseConnection()
