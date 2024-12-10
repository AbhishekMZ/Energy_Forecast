// Initialize main chart
document.addEventListener('DOMContentLoaded', function() {
    const ctx = document.getElementById('mainChart').getContext('2d');
    
    // API configuration
    const API_BASE_URL = 'http://localhost:8001/api';

    // Fetch cities from API
    async function fetchCities() {
        try {
            const response = await fetch(`${API_BASE_URL}/cities`);
            const cities = await response.json();
            updateCityDropdown(cities);
        } catch (error) {
            console.error('Error fetching cities:', error);
        }
    }

    // Update city dropdown
    function updateCityDropdown(cities) {
        const select = document.querySelector('select');
        select.innerHTML = cities.map(city => 
            `<option value="${city}">${city}</option>`
        ).join('');
    }

    // Fetch current usage
    async function fetchCurrentUsage(city) {
        try {
            const response = await fetch(`${API_BASE_URL}/current-usage/${city}`);
            const data = await response.json();
            updateCurrentUsage(data);
        } catch (error) {
            console.error('Error fetching current usage:', error);
        }
    }

    // Fetch forecast data
    async function fetchForecast(city, startDate, endDate) {
        try {
            const response = await fetch(
                `${API_BASE_URL}/forecast/${city}?start_date=${startDate}&end_date=${endDate}`
            );
            const data = await response.json();
            updateChart(data.forecast);
        } catch (error) {
            console.error('Error fetching forecast:', error);
        }
    }

    // Fetch efficiency metrics
    async function fetchEfficiency(city) {
        try {
            const response = await fetch(`${API_BASE_URL}/efficiency/${city}`);
            const data = await response.json();
            updateEfficiencyMetrics(data);
        } catch (error) {
            console.error('Error fetching efficiency metrics:', error);
        }
    }

    // Fetch recommendations
    async function fetchRecommendations(city) {
        try {
            const response = await fetch(`${API_BASE_URL}/recommendations/${city}`);
            const data = await response.json();
            updateRecommendations(data.recommendations);
        } catch (error) {
            console.error('Error fetching recommendations:', error);
        }
    }

    // Fetch all stats
    async function fetchStats(city) {
        try {
            const response = await fetch(`${API_BASE_URL}/stats/${city}`);
            const data = await response.json();
            updateStats(data);
        } catch (error) {
            console.error('Error fetching stats:', error);
        }
    }

    // Update current usage display
    function updateCurrentUsage(data) {
        document.querySelector('.text-xl').textContent = 
            `${data.current_usage.toFixed(1)} kWh`;
    }

    // Update chart with new data
    function updateChart(forecastData) {
        const chart = Chart.getChart('mainChart');
        if (chart) {
            chart.data.datasets[0].data = forecastData.map(d => d.predicted_consumption);
            chart.data.labels = forecastData.map(d => {
                const date = new Date(d.timestamp);
                return `${date.getHours()}:00`;
            });
            chart.update();
        }
    }

    // Update efficiency metrics
    function updateEfficiencyMetrics(data) {
        const efficiencyElements = document.querySelectorAll('.flex.justify-between');
        efficiencyElements[0].querySelector('.font-semibold').textContent = data.peak_hours.join(' - ');
        efficiencyElements[1].querySelector('.font-semibold').textContent = data.off_peak_hours.join(' - ');
    }

    // Update recommendations
    function updateRecommendations(recommendations) {
        const recommendationsContainer = document.querySelector('.recommendations .space-y-4');
        recommendationsContainer.innerHTML = recommendations.map(rec => `
            <div class="flex items-start">
                <div class="p-2 bg-green-100 rounded-full">
                    <i class="fas fa-lightbulb text-green-600"></i>
                </div>
                <div class="ml-4">
                    <p class="font-medium">${rec.title}</p>
                    <p class="text-sm text-gray-600">${rec.description}</p>
                    <p class="text-sm text-blue-600">Potential savings: ${rec.potential_savings}</p>
                </div>
            </div>
        `).join('');
    }

    // Update all stats
    function updateStats(data) {
        const statsElements = document.querySelectorAll('.text-xl.font-semibold');
        statsElements[0].textContent = `${data.current_usage.toFixed(1)} kWh`;
        statsElements[1].textContent = `${data.predicted_peak.toFixed(1)} kWh`;
        statsElements[2].textContent = `${data.efficiency.toFixed(1)}%`;
        statsElements[3].textContent = `${(data.accuracy * 100).toFixed(1)}%`;
    }

    // Initialize application
    fetchCities();
    
    // Set up event listeners
    const citySelect = document.querySelector('select');
    citySelect.addEventListener('change', function(e) {
        const city = e.target.value;
        updateDashboard(city);
    });
    
    const dateInputs = document.querySelectorAll('input[type="date"]');
    dateInputs.forEach(input => {
        input.addEventListener('change', function() {
            const city = citySelect.value;
            const startDate = dateInputs[0].value;
            const endDate = dateInputs[1].value;
            if (city && startDate && endDate) {
                fetchForecast(city, startDate, endDate);
            }
        });
    });

    // Update dashboard data
    async function updateDashboard(city) {
        // Show loading state
        document.querySelectorAll('.bg-white').forEach(el => {
            el.classList.add('loading');
        });
        
        // Fetch all data
        await Promise.all([
            fetchCurrentUsage(city),
            fetchStats(city),
            fetchEfficiency(city),
            fetchRecommendations(city)
        ]);
        
        // Get date range and fetch forecast
        const dateInputs = document.querySelectorAll('input[type="date"]');
        const startDate = dateInputs[0].value;
        const endDate = dateInputs[1].value;
        if (startDate && endDate) {
            await fetchForecast(city, startDate, endDate);
        }
        
        // Remove loading state
        document.querySelectorAll('.bg-white').forEach(el => {
            el.classList.remove('loading');
        });
    }

    // Sample data - replace with actual API data
    const data = {
        labels: Array.from({length: 24}, (_, i) => `${i}:00`),
        datasets: [
            {
                label: 'Actual Consumption',
                data: Array.from({length: 24}, () => Math.floor(Math.random() * 2000 + 2000)),
                borderColor: '#3B82F6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                fill: true
            },
            {
                label: 'Predicted Consumption',
                data: Array.from({length: 24}, () => Math.floor(Math.random() * 2000 + 2000)),
                borderColor: '#10B981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                fill: true
            }
        ]
    };

    const config = {
        type: 'line',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    position: 'top'
                },
                tooltip: {
                    enabled: true,
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Energy Consumption (kWh)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Time of Day'
                    }
                }
            }
        }
    };

    new Chart(ctx, config);
});

// Helper function to generate random data
function generateRandomData(count, min, max) {
    return Array.from({length: count}, () => 
        Math.floor(Math.random() * (max - min + 1)) + min
    );
}

// City selection handler
document.querySelector('select').addEventListener('change', function(e) {
    // Update data based on selected city
    updateDashboard(e.target.value);
});

// Date range handler
const dateInputs = document.querySelectorAll('input[type="date"]');
dateInputs.forEach(input => {
    input.addEventListener('change', function() {
        // Update data based on selected date range
        updateDateRange();
    });
});

// Update dashboard data
async function updateDashboard(city) {
    // Simulate loading state
    document.querySelectorAll('.bg-white').forEach(el => {
        el.classList.add('loading');
    });

    // Fetch new data from API
    setTimeout(() => {
        // Remove loading state
        document.querySelectorAll('.bg-white').forEach(el => {
            el.classList.remove('loading');
        });
        
        // Update stats and charts
        updateStats();
        updateCharts();
    }, 1000);
}

// Update statistics
function updateStats() {
    // Update current usage
    document.querySelector('.text-xl').textContent = 
        `${Math.floor(Math.random() * 1000 + 2000)} kWh`;
    
    // Update other stats
    // ... Add more stat updates
}

// Update charts
function updateCharts() {
    // Update main chart
    const chart = Chart.getChart('mainChart');
    if (chart) {
        chart.data.datasets.forEach(dataset => {
            dataset.data = generateRandomData(24, 2000, 3500);
        });
        chart.update();
    }
}

// Handle notifications
document.querySelector('.fa-bell').addEventListener('click', function() {
    // Show notifications panel
    showNotifications();
});

// Show notifications panel
function showNotifications() {
    // Implementation for notifications panel
    console.log('Show notifications');
}

// Initialize tooltips
const tooltips = document.querySelectorAll('[data-tooltip]');
tooltips.forEach(tooltip => {
    // Initialize tooltip functionality
    // ... Add tooltip initialization
});

// Handle responsive menu
const menuButton = document.querySelector('.menu-button');
if (menuButton) {
    menuButton.addEventListener('click', function() {
        // Toggle mobile menu
        document.querySelector('.md\\:flex').classList.toggle('hidden');
    });
}

// Error handling
window.addEventListener('error', function(e) {
    console.error('Application error:', e.error);
    // Implement error notification
});

// Cleanup function
window.addEventListener('unload', function() {
    // Cleanup resources
    // ... Add cleanup code
});
