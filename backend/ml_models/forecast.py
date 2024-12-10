import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

class EnergyForecaster:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def prepare_features(self, data):
        """Prepare features for the model"""
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime if string
        if isinstance(df['timestamp'].iloc[0], str):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Create features array
        features = df[[
            'hour', 'day_of_week', 'month', 'is_weekend',
            'temperature', 'humidity'
        ]].values
        
        return self.scaler.fit_transform(features)

    def train(self, historical_data):
        """Train the model with historical data"""
        features = self.prepare_features(historical_data)
        targets = np.array([d['consumption'] for d in historical_data])
        
        self.model.fit(features, targets)
        self.is_trained = True

    def predict(self, historical_data):
        """Generate predictions"""
        if not self.is_trained:
            self.train(historical_data)
        
        # Prepare future timestamps
        last_timestamp = max(d['timestamp'] for d in historical_data)
        future_timestamps = [
            last_timestamp + timedelta(hours=i)
            for i in range(1, 25)  # Next 24 hours
        ]
        
        # Create future features
        future_data = []
        for ts in future_timestamps:
            # Use last known temperature and humidity
            future_data.append({
                'timestamp': ts,
                'temperature': historical_data[-1]['temperature'],
                'humidity': historical_data[-1]['humidity'],
                'consumption': 0  # Placeholder
            })
        
        # Generate predictions
        future_features = self.prepare_features(future_data)
        predictions = self.model.predict(future_features)
        
        # Prepare response
        forecast_data = []
        for i, ts in enumerate(future_timestamps):
            forecast_data.append({
                'timestamp': ts.isoformat(),
                'predicted_consumption': float(predictions[i]),
                'confidence_level': float(self.model.score(
                    self.prepare_features(historical_data),
                    [d['consumption'] for d in historical_data]
                ))
            })
        
        return forecast_data

    def generate_recommendations(self, consumption_patterns):
        """Generate energy optimization recommendations"""
        df = pd.DataFrame(consumption_patterns)
        
        # Analyze patterns
        peak_consumption = df['consumption'].max()
        avg_consumption = df['consumption'].mean()
        peak_hours = df[df['consumption'] > avg_consumption * 1.2]
        
        recommendations = []
        
        # Generate recommendations based on patterns
        if len(peak_hours) > 0:
            recommendations.append({
                "type": "peak_reduction",
                "title": "Optimize Peak Usage",
                "description": "Shift non-essential operations to off-peak hours",
                "potential_savings": f"{((peak_consumption - avg_consumption) / peak_consumption * 100):.1f}%"
            })
        
        # Add general recommendations
        recommendations.extend([
            {
                "type": "temperature_control",
                "title": "Temperature Control",
                "description": "Adjust HVAC settings based on occupancy",
                "potential_savings": "10-15%"
            },
            {
                "type": "renewable_energy",
                "title": "Renewable Integration",
                "description": "Consider solar panel installation",
                "potential_savings": "20-30%"
            }
        ])
        
        return recommendations
