import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
import json
from datetime import datetime, timedelta

class DataVisualizer:
    """
    Advanced data visualization system for energy forecasting
    """
    def __init__(self):
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'tertiary': '#2ca02c',
            'quaternary': '#d62728',
            'background': '#ffffff',
            'grid': '#e6e6e6'
        }
        
    def create_dashboard(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """
        Create a complete dashboard with multiple visualizations
        """
        return {
            'time_series': self.plot_time_series(df),
            'correlations': self.plot_correlation_matrix(df),
            'distributions': self.plot_distributions(df),
            'patterns': self.plot_patterns(df),
            'anomalies': self.plot_anomalies(df)
        }
    
    def plot_time_series(self, df: pd.DataFrame) -> go.Figure:
        """
        Create interactive time series plot
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Energy Consumption Over Time', 'Weather Metrics'),
            vertical_spacing=0.12
        )
        
        # Energy consumption plot
        if 'total_load' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['total_load'],
                    name='Total Load',
                    line=dict(color=self.color_scheme['primary'])
                ),
                row=1, col=1
            )
            
        # Weather metrics
        weather_cols = ['temperature', 'humidity', 'wind_speed']
        for i, col in enumerate(weather_cols):
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col],
                        name=col.title(),
                        line=dict(color=list(self.color_scheme.values())[i+1])
                    ),
                    row=2, col=1
                )
                
        fig.update_layout(
            height=800,
            showlegend=True,
            template='plotly_white',
            title_text="Energy Consumption and Weather Metrics Analysis"
        )
        
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame) -> go.Figure:
        """
        Create interactive correlation matrix heatmap
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numerical_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            height=600,
            width=800,
            template='plotly_white'
        )
        
        return fig
    
    def plot_distributions(self, df: pd.DataFrame) -> go.Figure:
        """
        Create distribution plots for key variables
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        n_cols = len(numerical_cols)
        
        fig = make_subplots(
            rows=((n_cols-1)//2 + 1), cols=2,
            subplot_titles=[col.title() for col in numerical_cols]
        )
        
        for i, col in enumerate(numerical_cols):
            row = i//2 + 1
            col_num = i%2 + 1
            
            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=df[col],
                    name=col,
                    nbinsx=30,
                    marker_color=self.color_scheme['primary']
                ),
                row=row, col=col_num
            )
            
            # Add KDE
            kde = self._calculate_kde(df[col].dropna())
            fig.add_trace(
                go.Scatter(
                    x=kde['x'],
                    y=kde['y'],
                    name=f'{col} KDE',
                    line=dict(color=self.color_scheme['secondary'])
                ),
                row=row, col=col_num
            )
            
        fig.update_layout(
            height=300*((n_cols-1)//2 + 1),
            showlegend=False,
            template='plotly_white',
            title_text="Feature Distributions"
        )
        
        return fig
    
    def plot_patterns(self, df: pd.DataFrame) -> go.Figure:
        """
        Create pattern analysis plots
        """
        if 'total_load' not in df.columns or 'timestamp' not in df.columns:
            return None
            
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['dayofweek'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Hourly Pattern', 'Weekly Pattern')
        )
        
        # Hourly pattern
        hourly_pattern = df.groupby('hour')['total_load'].mean()
        fig.add_trace(
            go.Scatter(
                x=hourly_pattern.index,
                y=hourly_pattern.values,
                name='Hourly Pattern',
                line=dict(color=self.color_scheme['primary'])
            ),
            row=1, col=1
        )
        
        # Weekly pattern
        weekly_pattern = df.groupby('dayofweek')['total_load'].mean()
        fig.add_trace(
            go.Scatter(
                x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                y=weekly_pattern.values,
                name='Weekly Pattern',
                line=dict(color=self.color_scheme['secondary'])
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            template='plotly_white',
            title_text="Energy Consumption Patterns"
        )
        
        return fig
    
    def plot_anomalies(self, df: pd.DataFrame) -> go.Figure:
        """
        Create anomaly detection plot
        """
        if 'total_load' not in df.columns:
            return None
            
        # Calculate rolling statistics
        rolling_mean = df['total_load'].rolling(window=24).mean()
        rolling_std = df['total_load'].rolling(window=24).std()
        
        # Detect anomalies
        anomalies = df[abs(df['total_load'] - rolling_mean) > 3 * rolling_std]
        
        fig = go.Figure()
        
        # Plot original data
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['total_load'],
                name='Total Load',
                line=dict(color=self.color_scheme['primary'])
            )
        )
        
        # Plot anomalies
        fig.add_trace(
            go.Scatter(
                x=anomalies.index,
                y=anomalies['total_load'],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    color=self.color_scheme['quaternary'],
                    size=10,
                    symbol='circle'
                )
            )
        )
        
        fig.update_layout(
            height=500,
            template='plotly_white',
            title_text="Anomaly Detection in Energy Consumption"
        )
        
        return fig
    
    def _calculate_kde(self, data: pd.Series) -> Dict[str, np.ndarray]:
        """
        Calculate Kernel Density Estimation
        """
        from scipy import stats
        
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(min(data), max(data), 100)
        return {'x': x_range, 'y': kde(x_range)}
    
    def save_dashboard(self, figures: Dict[str, go.Figure], output_dir: str):
        """
        Save dashboard figures to HTML files
        """
        for name, fig in figures.items():
            if fig is not None:
                fig.write_html(f"{output_dir}/{name}.html")
                
    def create_report(self, df: pd.DataFrame, output_file: str):
        """
        Create and save a comprehensive visual report
        """
        dashboard = self.create_dashboard(df)
        
        # Create report HTML
        html_content = """
        <html>
        <head>
            <title>Energy Consumption Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .plot-container { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
                h1, h2 { color: #333; }
            </style>
        </head>
        <body>
            <h1>Energy Consumption Analysis Report</h1>
        """
        
        # Add each plot
        for name, fig in dashboard.items():
            if fig is not None:
                html_content += f"""
                <div class="plot-container">
                    <h2>{name.replace('_', ' ').title()}</h2>
                    {fig.to_html(full_html=False)}
                </div>
                """
                
        html_content += """
        </body>
        </html>
        """
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(html_content)
