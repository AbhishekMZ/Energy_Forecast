"""Initial database schema migration

This migration creates the core tables for the energy forecasting system.
"""

from alembic import op
import sqlalchemy as sa
from datetime import datetime

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create energy consumption table
    op.create_table(
        'energy_consumption',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('city', sa.String(100), nullable=False),
        sa.Column('consumption_kwh', sa.Float, nullable=False),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime, onupdate=datetime.utcnow)
    )
    
    # Create model versions table
    op.create_table(
        'model_versions',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('model_path', sa.String(255), nullable=False),
        sa.Column('metrics', sa.JSON, nullable=True),
        sa.Column('is_active', sa.Boolean, default=False),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow)
    )
    
    # Create feature store table
    op.create_table(
        'feature_store',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('feature_name', sa.String(100), nullable=False),
        sa.Column('feature_value', sa.Float, nullable=False),
        sa.Column('city', sa.String(100), nullable=False),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow)
    )
    
    # Create experiment tracking table
    op.create_table(
        'experiments',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('experiment_name', sa.String(100), nullable=False),
        sa.Column('model_version_id', sa.Integer, sa.ForeignKey('model_versions.id')),
        sa.Column('parameters', sa.JSON, nullable=True),
        sa.Column('metrics', sa.JSON, nullable=True),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime, onupdate=datetime.utcnow)
    )

def downgrade():
    op.drop_table('experiments')
    op.drop_table('feature_store')
    op.drop_table('model_versions')
    op.drop_table('energy_consumption')
