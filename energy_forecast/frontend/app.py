"""Streamlit interface for energy forecasting visualization"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from energy_forecast.core.visualization.plots import (
    plot_energy_demand_over_time,
    plot_weather_correlation,
    plot_daily_pattern,
    plot_forecast_comparison,
    plot_model_metrics
)
from energy_forecast.core.models.base_model import BaseModel
from energy_forecast.config.constants import CITIES

def load_data():
    """Load and preprocess data"""
    # This is a placeholder - replace with actual data loading logic
    pass

def main():
    st.set_page_config(
        page_title="Energy Forecast Dashboard",
        page_icon="⚡",
        layout="wide"
    )
    
    # Sidebar
    st.sidebar.title("⚡ Energy Forecast")
    
    # City selection
    city = st.sidebar.selectbox(
        "Select City",
        options=list(CITIES.keys())
    )
    
    # Date range selection
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(datetime.now() - timedelta(days=30), datetime.now()),
        max_value=datetime.now()
    )
    
    # Load data
    data = load_data()  # Replace with actual data
    
    # Main content
    st.title(f"Energy Forecast Dashboard - {city}")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Current Demand",
            "3,500 MW",
            "50 MW"
        )
    with col2:
        st.metric(
            "Temperature",
            "32°C",
            "2°C"
        )
    with col3:
        st.metric(
            "Solar Generation",
            "800 MW",
            "-100 MW"
        )
    with col4:
        st.metric(
            "Grid Load",
            "75%",
            "5%"
        )
    
    # Tabs for different visualizations
    tab1, tab2, tab3 = st.tabs([
        "Demand Analysis",
        "Weather Impact",
        "Model Performance"
    ])
    
    with tab1:
        st.subheader("Energy Demand Analysis")
        
        # Time series plot
        if data is not None:
            fig = plot_energy_demand_over_time(data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Daily pattern
            fig = plot_daily_pattern(data)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Weather Impact Analysis")
        
        if data is not None:
            # Weather correlation plot
            fig = plot_weather_correlation(data)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Model Performance")
        
        if data is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                # Forecast comparison
                fig = plot_forecast_comparison(
                    data['total_demand'],
                    data['predicted_demand']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Model metrics
                metrics = {
                    'r2_score': 0.85,
                    'rmse': 250,
                    'mae': 200
                }
                fig = plot_model_metrics(metrics)
                st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Energy Forecast Dashboard | Created with Streamlit and Plotly</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
