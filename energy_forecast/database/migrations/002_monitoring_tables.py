"""Add monitoring and experiment tracking tables.

This migration adds tables for enhanced monitoring and A/B testing.
"""

from alembic import op
import sqlalchemy as sa
from datetime import datetime

# revision identifiers
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None

def upgrade():
    # Create performance metrics table
    op.create_table(
        'performance_metrics',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('metric_name', sa.String(100), nullable=False),
        sa.Column('metric_value', sa.Float, nullable=False),
        sa.Column('metric_type', sa.String(50), nullable=False),
        sa.Column('tags', sa.JSON, nullable=True),
        sa.Column('timestamp', sa.DateTime, default=datetime.utcnow),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow)
    )
    
    # Create error logs table
    op.create_table(
        'error_logs',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('error_type', sa.String(100), nullable=False),
        sa.Column('error_message', sa.Text, nullable=False),
        sa.Column('stack_trace', sa.Text, nullable=True),
        sa.Column('context', sa.JSON, nullable=True),
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow)
    )
    
    # Create experiment events table
    op.create_table(
        'experiment_events',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('experiment_id', sa.String(100), nullable=False),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('event_data', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow)
    )
    
    # Create model performance table
    op.create_table(
        'model_performance',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('model_version_id', sa.Integer, sa.ForeignKey('model_versions.id')),
        sa.Column('metric_name', sa.String(100), nullable=False),
        sa.Column('metric_value', sa.Float, nullable=False),
        sa.Column('prediction_id', sa.String(100), nullable=True),
        sa.Column('actual_value', sa.Float, nullable=True),
        sa.Column('predicted_value', sa.Float, nullable=True),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow)
    )
    
    # Create system health table
    op.create_table(
        'system_health',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('component', sa.String(100), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('details', sa.JSON, nullable=True),
        sa.Column('last_check', sa.DateTime, nullable=False),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow)
    )
    
    # Create cache metrics table
    op.create_table(
        'cache_metrics',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('cache_key', sa.String(255), nullable=False),
        sa.Column('hit_count', sa.Integer, default=0),
        sa.Column('miss_count', sa.Integer, default=0),
        sa.Column('last_accessed', sa.DateTime, nullable=True),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime, onupdate=datetime.utcnow)
    )
    
    # Create batch processing metrics table
    op.create_table(
        'batch_metrics',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('batch_id', sa.String(100), nullable=False),
        sa.Column('batch_size', sa.Integer, nullable=False),
        sa.Column('processing_time', sa.Float, nullable=False),
        sa.Column('success_count', sa.Integer, nullable=False),
        sa.Column('error_count', sa.Integer, nullable=False),
        sa.Column('start_time', sa.DateTime, nullable=False),
        sa.Column('end_time', sa.DateTime, nullable=False),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow)
    )

def downgrade():
    op.drop_table('batch_metrics')
    op.drop_table('cache_metrics')
    op.drop_table('system_health')
    op.drop_table('model_performance')
    op.drop_table('experiment_events')
    op.drop_table('error_logs')
    op.drop_table('performance_metrics')
