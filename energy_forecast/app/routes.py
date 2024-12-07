from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import pandas as pd
import json

from database.operations import DatabaseManager
from models.prediction.predictor import EnergyPredictor
from utils.data_processor import DataProcessor

app = Flask(__name__)
db = DatabaseManager()
predictor = EnergyPredictor()
data_processor = DataProcessor()

@app.route('/')
def index():
    """Home page with dashboard"""
    return render_template('index.html')

@app.route('/data/energy')
def get_energy_data():
    """Get energy consumption data for visualization"""
    days = int(request.args.get('days', 7))
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = db.get_energy_consumption(start_date, end_date)
    return jsonify([{
        'timestamp': item.timestamp.isoformat(),
        'total_load': item.total_load,
        'fossil_fuel': item.fossil_fuel,
        'hydro': item.hydro,
        'nuclear': item.nuclear,
        'solar': item.solar,
        'wind': item.wind
    } for item in data])

@app.route('/data/weather')
def get_weather_data():
    """Get weather data for visualization"""
    city_id = int(request.args.get('city_id'))
    days = int(request.args.get('days', 7))
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = db.get_weather_data(city_id, start_date, end_date)
    return jsonify([{
        'timestamp': item.timestamp.isoformat(),
        'temperature': item.temperature,
        'pressure': item.pressure,
        'humidity': item.humidity,
        'wind_speed': item.wind_speed
    } for item in data])

@app.route('/predict', methods=['POST'])
def predict():
    """Make energy consumption predictions"""
    try:
        data = request.get_json()
        prediction = predictor.predict(data)
        
        # Store prediction in database
        db.add_prediction({
            'timestamp': datetime.now(),
            'predicted_load': prediction['value'],
            'model_name': prediction['model_name']
        })
        
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model/performance')
def model_performance():
    """Get model performance metrics"""
    model_name = request.args.get('model_name', 'default_model')
    metrics = db.get_model_performance(model_name)
    
    if metrics:
        return jsonify({
            'mae': metrics.mae,
            'mse': metrics.mse,
            'r2_score': metrics.r2_score,
            'training_date': metrics.training_date.isoformat()
        })
    return jsonify({'error': 'No metrics found'}), 404

@app.route('/data/upload', methods=['POST'])
def upload_data():
    """Upload new data to the system"""
    try:
        file = request.files['file']
        if file:
            # Process the uploaded file
            df = pd.read_csv(file)
            processed_data = data_processor.process(df)
            
            # Store in database
            for record in processed_data:
                db.add_energy_consumption(record)
            
            return jsonify({'message': 'Data uploaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/dashboard/summary')
def dashboard_summary():
    """Get summary statistics for dashboard"""
    try:
        # Get recent predictions
        predictions = db.get_recent_predictions(hours=24)
        
        # Get latest weather data
        weather = db.get_weather_data(
            city_id=1,  # Default city
            start_date=datetime.now() - timedelta(hours=24),
            end_date=datetime.now()
        )
        
        return jsonify({
            'predictions': len(predictions),
            'average_load': sum(p.predicted_load for p in predictions) / len(predictions) if predictions else 0,
            'weather_records': len(weather)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
