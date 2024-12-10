import os

# Create necessary directories
directories = [
    'backend',
    'backend/models',
    'backend/ml_models',
    'frontend/src',
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
