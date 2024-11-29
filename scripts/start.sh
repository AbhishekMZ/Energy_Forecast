#!/bin/bash

# Wait for database to be ready
echo "Waiting for database..."
while ! nc -z db 5432; do
  sleep 1
done
echo "Database is ready!"

# Run migrations
echo "Running database migrations..."
alembic upgrade head

# Create admin user if not exists
echo "Initializing admin user..."
python scripts/init_admin.py

# Start the application
echo "Starting the application..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
