from setuptools import setup, find_packages

setup(
    name="energy_forecast",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "pytest",
        "flask",
        "sqlalchemy",
        "alembic",
        "scikit-learn",
        "plotly",
        "dash",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive energy consumption forecasting system for Indian cities",
    long_description=open("energy_forecast/README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
)
