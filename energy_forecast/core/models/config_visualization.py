from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
import warnings
from .config_validation import ValidationResult

class ConfigurationVisualizer:
    """Visualize model configurations and their relationships with performance"""
    
    def __init__(self, style: str = 'plotly'):
        """
        Initialize visualizer
        
        Parameters:
            style: Visualization style ('plotly' or 'matplotlib')
        """
        self.style = style
        self.color_palette = px.colors.qualitative.Set3
        plt.style.use('seaborn')
    
    def plot_performance_comparison(self,
                                  results: List[ValidationResult],
                                  metric: str = 'rmse',
                                  show_variance: bool = True) -> go.Figure:
        """Plot performance comparison across models"""
        models = [r.model_name for r in results]
        mean_metrics = [r.metrics[metric] for r in results]
        std_metrics = [np.std([m[metric] for m in r.fold_metrics]) for r in results]
        
        if self.style == 'plotly':
            fig = go.Figure()
            
            # Add bar chart for mean performance
            fig.add_trace(go.Bar(
                name='Mean Performance',
                x=models,
                y=mean_metrics,
                error_y=dict(
                    type='data',
                    array=std_metrics if show_variance else None,
                    visible=show_variance
                )
            ))
            
            fig.update_layout(
                title=f'Model Performance Comparison ({metric.upper()})',
                xaxis_title='Model',
                yaxis_title=metric.upper(),
                template='plotly_white'
            )
            
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(models, mean_metrics, yerr=std_metrics if show_variance else None)
            ax.set_title(f'Model Performance Comparison ({metric.upper()})')
            ax.set_xlabel('Model')
            ax.set_ylabel(metric.upper())
            plt.xticks(rotation=45)
            plt.tight_layout()
            
        return fig
    
    def plot_parameter_importance(self,
                                results: List[ValidationResult],
                                top_n: int = 10) -> go.Figure:
        """Visualize parameter importance across models"""
        # Extract parameter values and corresponding performance
        param_importance = {}
        
        for result in results:
            for param, value in result.config.items():
                if isinstance(value, (int, float)):
                    if param not in param_importance:
                        param_importance[param] = {
                            'values': [],
                            'performance': []
                        }
                    param_importance[param]['values'].append(value)
                    param_importance[param]['performance'].append(result.mean_rmse)
        
        # Calculate correlation between parameters and performance
        correlations = {}
        for param, data in param_importance.items():
            if len(set(data['values'])) > 1:  # Only if parameter varies
                corr, _ = spearmanr(data['values'], data['performance'])
                correlations[param] = abs(corr)
        
        # Get top N most important parameters
        top_params = dict(sorted(correlations.items(),
                               key=lambda x: abs(x[1]),
                               reverse=True)[:top_n])
        
        if self.style == 'plotly':
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=list(top_params.keys()),
                y=list(top_params.values()),
                marker_color=self.color_palette
            ))
            
            fig.update_layout(
                title='Parameter Importance (Correlation with Performance)',
                xaxis_title='Parameter',
                yaxis_title='Absolute Correlation',
                template='plotly_white'
            )
            
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(top_params.keys(), top_params.values())
            ax.set_title('Parameter Importance (Correlation with Performance)')
            ax.set_xlabel('Parameter')
            ax.set_ylabel('Absolute Correlation')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
        return fig
    
    def plot_parameter_relationships(self,
                                   results: List[ValidationResult]) -> go.Figure:
        """Visualize relationships between parameters and performance"""
        # Extract numeric parameters
        params_data = []
        for result in results:
            row = {'model': result.model_name, 'rmse': result.mean_rmse}
            row.update({k: v for k, v in result.config.items()
                       if isinstance(v, (int, float))})
            params_data.append(row)
        
        df = pd.DataFrame(params_data)
        
        if self.style == 'plotly':
            fig = make_subplots(
                rows=len(df.columns) - 2,
                cols=1,
                subplot_titles=[f'{col} vs RMSE'
                              for col in df.columns if col not in ['model', 'rmse']]
            )
            
            for i, param in enumerate(df.columns):
                if param not in ['model', 'rmse']:
                    fig.add_trace(
                        go.Scatter(
                            x=df[param],
                            y=df['rmse'],
                            mode='markers',
                            name=param,
                            marker=dict(
                                color=df['rmse'],
                                colorscale='Viridis',
                                showscale=True
                            )
                        ),
                        row=i+1,
                        col=1
                    )
            
            fig.update_layout(
                height=300 * (len(df.columns) - 2),
                title='Parameter vs Performance Relationships',
                showlegend=False,
                template='plotly_white'
            )
            
        else:
            n_params = len(df.columns) - 2
            fig, axes = plt.subplots(n_params, 1, figsize=(10, 5*n_params))
            
            for i, param in enumerate(df.columns):
                if param not in ['model', 'rmse']:
                    axes[i].scatter(df[param], df['rmse'])
                    axes[i].set_xlabel(param)
                    axes[i].set_ylabel('RMSE')
                    axes[i].set_title(f'{param} vs RMSE')
            
            plt.tight_layout()
            
        return fig
    
    def plot_configuration_clusters(self,
                                  results: List[ValidationResult]) -> go.Figure:
        """Visualize clusters of similar configurations"""
        # Extract numeric parameters
        config_data = []
        for result in results:
            row = {k: v for k, v in result.config.items()
                  if isinstance(v, (int, float))}
            config_data.append(row)
        
        df = pd.DataFrame(config_data)
        
        # Normalize data
        normalized_df = (df - df.mean()) / df.std()
        
        # Calculate distance matrix
        distance_matrix = hierarchy.distance.pdist(normalized_df)
        linkage_matrix = hierarchy.linkage(distance_matrix, method='ward')
        
        if self.style == 'plotly':
            fig = ff.create_dendrogram(
                normalized_df,
                labels=[r.model_name for r in results],
                orientation='left',
                colorscale=self.color_palette
            )
            
            fig.update_layout(
                title='Configuration Similarity Dendrogram',
                width=800,
                height=600,
                template='plotly_white'
            )
            
        else:
            fig, ax = plt.subplots(figsize=(10, 7))
            hierarchy.dendrogram(
                linkage_matrix,
                labels=[r.model_name for r in results],
                leaf_rotation=0
            )
            ax.set_title('Configuration Similarity Dendrogram')
            plt.tight_layout()
            
        return fig
    
    def plot_validation_stability(self,
                                results: List[ValidationResult]) -> go.Figure:
        """Visualize validation stability across folds"""
        models = []
        fold_performances = []
        fold_numbers = []
        
        for result in results:
            for fold_idx, metrics in enumerate(result.fold_metrics):
                models.append(result.model_name)
                fold_performances.append(metrics['rmse'])
                fold_numbers.append(f'Fold {fold_idx + 1}')
        
        if self.style == 'plotly':
            fig = go.Figure()
            
            for model in set(models):
                mask = [m == model for m in models]
                fig.add_trace(go.Box(
                    y=[p for p, m in zip(fold_performances, mask) if m],
                    name=model,
                    boxpoints='all'
                ))
            
            fig.update_layout(
                title='Validation Stability Across Folds',
                yaxis_title='RMSE',
                template='plotly_white',
                showlegend=True
            )
            
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=models, y=fold_performances)
            ax.set_title('Validation Stability Across Folds')
            ax.set_xlabel('Model')
            ax.set_ylabel('RMSE')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
        return fig
    
    def plot_configuration_evolution(self,
                                   results: List[ValidationResult]) -> go.Figure:
        """Visualize how configurations evolved during validation"""
        # Extract parameter changes
        param_evolution = {}
        
        for result in results:
            for param, value in result.config.items():
                if isinstance(value, (int, float)):
                    if param not in param_evolution:
                        param_evolution[param] = []
                    param_evolution[param].append(value)
        
        if self.style == 'plotly':
            fig = go.Figure()
            
            for param, values in param_evolution.items():
                fig.add_trace(go.Scatter(
                    y=values,
                    name=param,
                    mode='lines+markers'
                ))
            
            fig.update_layout(
                title='Configuration Parameter Evolution',
                xaxis_title='Configuration Index',
                yaxis_title='Parameter Value (Normalized)',
                template='plotly_white'
            )
            
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            for param, values in param_evolution.items():
                ax.plot(values, label=param, marker='o')
            ax.set_title('Configuration Parameter Evolution')
            ax.set_xlabel('Configuration Index')
            ax.set_ylabel('Parameter Value (Normalized)')
            ax.legend()
            plt.tight_layout()
            
        return fig
    
    def create_dashboard(self,
                        results: List[ValidationResult]) -> go.Figure:
        """Create comprehensive dashboard of configuration analysis"""
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                'Model Performance',
                'Parameter Importance',
                'Parameter Relationships',
                'Configuration Clusters',
                'Validation Stability',
                'Configuration Evolution'
            )
        )
        
        # Add all plots to dashboard
        perf_fig = self.plot_performance_comparison(results)
        param_imp_fig = self.plot_parameter_importance(results)
        param_rel_fig = self.plot_parameter_relationships(results)
        cluster_fig = self.plot_configuration_clusters(results)
        stab_fig = self.plot_validation_stability(results)
        evol_fig = self.plot_configuration_evolution(results)
        
        # Combine all figures
        for trace in perf_fig.data:
            fig.add_trace(trace, row=1, col=1)
        for trace in param_imp_fig.data:
            fig.add_trace(trace, row=1, col=2)
        for trace in param_rel_fig.data:
            fig.add_trace(trace, row=2, col=1)
        for trace in cluster_fig.data:
            fig.add_trace(trace, row=2, col=2)
        for trace in stab_fig.data:
            fig.add_trace(trace, row=3, col=1)
        for trace in evol_fig.data:
            fig.add_trace(trace, row=3, col=2)
        
        fig.update_layout(
            height=1800,
            width=1200,
            title='Configuration Analysis Dashboard',
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def save_visualization(self, fig: go.Figure,
                          filename: str,
                          format: str = 'html') -> None:
        """Save visualization to file"""
        if format == 'html':
            fig.write_html(filename)
        else:
            fig.write_image(filename)
