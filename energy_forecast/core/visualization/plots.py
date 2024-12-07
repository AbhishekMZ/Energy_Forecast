"""Interactive visualization module for energy forecasting data"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime, timedelta

def plot_energy_demand_over_time(data: pd.DataFrame) -> go.Figure:
    """Create an interactive time series plot of energy demand"""
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['total_demand'],
            name='Total Demand',
            mode='lines',
            line=dict(color='#1f77b4'),
            hovertemplate='%{y:.2f} MW<br>%{x}<extra></extra>'
        )
    )
    
    fig.update_layout(
        title='Energy Demand Over Time',
        xaxis_title='Date',
        yaxis_title='Energy Demand (MW)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def plot_weather_correlation(data: pd.DataFrame) -> go.Figure:
    """Create an interactive correlation plot between weather and demand"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Temperature vs Demand',
            'Solar Radiation vs Demand',
            'Humidity vs Demand',
            'Cloud Cover vs Demand'
        )
    )
    
    # Temperature vs Demand
    fig.add_trace(
        go.Scatter(
            x=data['temperature'],
            y=data['total_demand'],
            mode='markers',
            name='Temperature',
            marker=dict(color='#1f77b4', size=5),
            hovertemplate='Temp: %{x:.1f}°C<br>Demand: %{y:.0f} MW<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Solar Radiation vs Demand
    fig.add_trace(
        go.Scatter(
            x=data['solar_radiation'],
            y=data['total_demand'],
            mode='markers',
            name='Solar',
            marker=dict(color='#ff7f0e', size=5),
            hovertemplate='Solar: %{x:.1f} W/m²<br>Demand: %{y:.0f} MW<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Humidity vs Demand
    fig.add_trace(
        go.Scatter(
            x=data['humidity'],
            y=data['total_demand'],
            mode='markers',
            name='Humidity',
            marker=dict(color='#2ca02c', size=5),
            hovertemplate='Humidity: %{x:.1f}%<br>Demand: %{y:.0f} MW<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Cloud Cover vs Demand
    fig.add_trace(
        go.Scatter(
            x=data['cloud_cover'],
            y=data['total_demand'],
            mode='markers',
            name='Cloud Cover',
            marker=dict(color='#d62728', size=5),
            hovertemplate='Cloud: %{x:.1f}%<br>Demand: %{y:.0f} MW<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title='Weather Parameters vs Energy Demand',
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

def plot_daily_pattern(data: pd.DataFrame) -> go.Figure:
    """Create an interactive plot showing daily demand patterns"""
    data['hour'] = data.index.hour
    hourly_avg = data.groupby('hour')['total_demand'].mean()
    hourly_std = data.groupby('hour')['total_demand'].std()
    
    fig = go.Figure()
    
    # Add mean line
    fig.add_trace(
        go.Scatter(
            x=hourly_avg.index,
            y=hourly_avg.values,
            name='Average Demand',
            line=dict(color='#1f77b4'),
            hovertemplate='Hour: %{x}<br>Avg Demand: %{y:.0f} MW<extra></extra>'
        )
    )
    
    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=hourly_avg.index.tolist() + hourly_avg.index.tolist()[::-1],
            y=(hourly_avg + hourly_std).tolist() + (hourly_avg - hourly_std).tolist()[::-1],
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=False
        )
    )
    
    fig.update_layout(
        title='Daily Demand Pattern',
        xaxis_title='Hour of Day',
        yaxis_title='Energy Demand (MW)',
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),
        template='plotly_white'
    )
    
    return fig

def plot_forecast_comparison(actual: pd.Series, predicted: pd.Series) -> go.Figure:
    """Create an interactive plot comparing actual vs predicted values"""
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(
        go.Scatter(
            x=actual.index,
            y=actual.values,
            name='Actual',
            line=dict(color='#1f77b4'),
            hovertemplate='%{y:.2f} MW<br>%{x}<extra>Actual</extra>'
        )
    )
    
    # Add predicted values
    fig.add_trace(
        go.Scatter(
            x=predicted.index,
            y=predicted.values,
            name='Predicted',
            line=dict(color='#ff7f0e', dash='dot'),
            hovertemplate='%{y:.2f} MW<br>%{x}<extra>Predicted</extra>'
        )
    )
    
    fig.update_layout(
        title='Forecast Comparison',
        xaxis_title='Date',
        yaxis_title='Energy Demand (MW)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def plot_model_metrics(metrics: Dict[str, float]) -> go.Figure:
    """Create an interactive gauge chart for model metrics"""
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]]
    )
    
    # R² Score
    fig.add_trace(
        go.Indicator(
            mode='gauge+number',
            value=metrics.get('r2_score', 0),
            title={'text': 'R² Score'},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': '#1f77b4'},
                'steps': [
                    {'range': [0, 0.5], 'color': '#ff7f7f'},
                    {'range': [0.5, 0.7], 'color': '#ffbf7f'},
                    {'range': [0.7, 1], 'color': '#7fbf7f'}
                ]
            }
        ),
        row=1, col=1
    )
    
    # RMSE
    max_rmse = 500  # From MODEL_THRESHOLDS
    fig.add_trace(
        go.Indicator(
            mode='gauge+number',
            value=metrics.get('rmse', 0),
            title={'text': 'RMSE'},
            gauge={
                'axis': {'range': [0, max_rmse]},
                'bar': {'color': '#1f77b4'},
                'steps': [
                    {'range': [0, 200], 'color': '#7fbf7f'},
                    {'range': [200, 350], 'color': '#ffbf7f'},
                    {'range': [350, max_rmse], 'color': '#ff7f7f'}
                ]
            }
        ),
        row=1, col=2
    )
    
    # MAE
    fig.add_trace(
        go.Indicator(
            mode='gauge+number',
            value=metrics.get('mae', 0),
            title={'text': 'MAE'},
            gauge={
                'axis': {'range': [0, max_rmse]},
                'bar': {'color': '#1f77b4'},
                'steps': [
                    {'range': [0, 150], 'color': '#7fbf7f'},
                    {'range': [150, 300], 'color': '#ffbf7f'},
                    {'range': [300, max_rmse], 'color': '#ff7f7f'}
                ]
            }
        ),
        row=1, col=3
    )
    
    fig.update_layout(
        title='Model Performance Metrics',
        height=400,
        template='plotly_white'
    )
    
    return fig
