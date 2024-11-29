# Energy Consumption Forecasting System

An end-to-end machine learning system for predicting energy consumption based on weather data.

## Project Structure

```
energy_forecast/
├── app/                    # Web application
│   ├── static/            # Static files (CSS, JS)
│   ├── templates/         # HTML templates
│   └── routes.py          # Flask routes
├── data/                  # Data files
│   ├── raw/              # Raw data
│   └── processed/        # Processed data
├── models/               # ML models
│   ├── training/        # Model training scripts
│   └── prediction/      # Prediction scripts
├── database/            # Database related files
│   ├── models.py       # SQLAlchemy models
│   └── operations.py   # Database operations
├── utils/              # Utility functions
├── tests/             # Unit tests
├── config.py         # Configuration
└── requirements.txt  # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the database:
```bash
# Configure your database settings in config.py
python database/setup.py
```

4. Run the application:
```bash
python run.py
```

## Features

- Real-time energy consumption monitoring
- Weather-based energy prediction
- Interactive data visualization
- Historical data analysis
- Multiple ML models (Polynomial Regression, LSTM, XGBoost)
- API endpoints for predictions
- Database integration for data persistence

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
