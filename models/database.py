from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, DateTime, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import enum
from datetime import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./energy_forecast.db"
# For production, use PostgreSQL:
# SQLALCHEMY_DATABASE_URL = "postgresql://user:password@localhost/energy_forecast"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class SupplierType(str, enum.Enum):
    SOLAR = "solar"
    WIND = "wind"
    HYDRO = "hydro"
    THERMAL = "thermal"
    NUCLEAR = "nuclear"
    OTHER = "other"

class PowerSupplier(Base):
    __tablename__ = "power_suppliers"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    company_name = Column(String)
    supplier_type = Column(Enum(SupplierType))
    license_number = Column(String, unique=True)
    capacity_mw = Column(Float)  # Installed capacity in Megawatts
    location = Column(String)  # City/Region
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)  # Admin verification status
    role = Column(String, default="supplier")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Create all tables
Base.metadata.create_all(bind=engine)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
