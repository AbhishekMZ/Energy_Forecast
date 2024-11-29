// Dashboard initialization
document.addEventListener('DOMContentLoaded', function() {
    updateDashboard();
    setInterval(updateDashboard, 300000); // Update every 5 minutes
});

// Update dashboard data
async function updateDashboard() {
    try {
        const summary = await fetchDashboardSummary();
        updateSummaryCards(summary);
        
        const energyData = await fetchEnergyData();
        updateEnergyChart(energyData);
        
        const weatherData = await fetchWeatherData();
        updateWeatherInfo(weatherData);
        
        const modelPerformance = await fetchModelPerformance();
        updateModelMetrics(modelPerformance);
    } catch (error) {
        console.error('Error updating dashboard:', error);
    }
}

// Fetch data from API endpoints
async function fetchDashboardSummary() {
    const response = await fetch('/dashboard/summary');
    return await response.json();
}

async function fetchEnergyData() {
    const response = await fetch('/data/energy?days=7');
    return await response.json();
}

async function fetchWeatherData() {
    const response = await fetch('/data/weather?city_id=1&days=1');
    return await response.json();
}

async function fetchModelPerformance() {
    const response = await fetch('/model/performance');
    return await response.json();
}

// Update UI elements
function updateSummaryCards(summary) {
    document.getElementById('current-load').textContent = 
        `${Math.round(summary.average_load)} MW`;
    document.getElementById('prediction').textContent = 
        `${summary.predictions} predictions`;
}

function updateEnergyChart(data) {
    const timestamps = data.map(d => d.timestamp);
    const loads = data.map(d => d.total_load);
    
    const trace = {
        x: timestamps,
        y: loads,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Energy Consumption'
    };
    
    const layout = {
        title: 'Energy Consumption Over Time',
        xaxis: { title: 'Time' },
        yaxis: { title: 'Load (MW)' }
    };
    
    Plotly.newPlot('energy-chart', [trace], layout);
}

function updateWeatherInfo(data) {
    if (data.length > 0) {
        const latest = data[data.length - 1];
        document.getElementById('temperature').textContent = 
            `${latest.temperature}°C`;
        document.getElementById('humidity').textContent = 
            `${latest.humidity}%`;
        document.getElementById('wind-speed').textContent = 
            `${latest.wind_speed} m/s`;
    }
}

function updateModelMetrics(metrics) {
    document.getElementById('accuracy').textContent = 
        `${(metrics.r2_score * 100).toFixed(1)}%`;
}

// Handle form submissions
document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    const date = document.getElementById('prediction-date').value;
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ date: date })
        });
        
        const result = await response.json();
        alert(`Predicted load: ${result.value} MW`);
    } catch (error) {
        console.error('Error making prediction:', error);
        alert('Error making prediction. Please try again.');
    }
});

document.getElementById('upload-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    const file = document.getElementById('data-file').files[0];
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/data/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        alert(result.message);
        updateDashboard(); // Refresh dashboard after upload
    } catch (error) {
        console.error('Error uploading data:', error);
        alert('Error uploading data. Please try again.');
    }
});
