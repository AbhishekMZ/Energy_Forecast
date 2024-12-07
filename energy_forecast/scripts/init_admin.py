from sqlalchemy.orm import Session
from models.database import get_db, PowerSupplier, engine
from auth.security import get_password_hash

def create_admin_user(db: Session):
    # Check if admin already exists
    admin = db.query(PowerSupplier).filter(PowerSupplier.username == "admin").first()
    if admin:
        print("Admin user already exists")
        return

    # Create admin user
    admin_user = PowerSupplier(
        username="admin",
        email="admin@energyforecast.com",
        full_name="System Administrator",
        company_name="Energy Forecast System",
        supplier_type="other",
        license_number="ADMIN001",
        capacity_mw=0.0,
        location="System",
        hashed_password=get_password_hash("admin123"),
        is_active=True,
        is_verified=True,
        role="admin"
    )

    try:
        db.add(admin_user)
        db.commit()
        print("Admin user created successfully")
    except Exception as e:
        db.rollback()
        print(f"Error creating admin user: {str(e)}")

if __name__ == "__main__":
    # Get database session
    db = next(get_db())
    create_admin_user(db)
