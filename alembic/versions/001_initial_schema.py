"""Initial schema

Revision ID: 001
Revises: 
Create Date: 2024-12-10 23:15:09.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create cities table
    op.create_table(
        'cities',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('region', sa.String(), nullable=True),
        sa.Column('population', sa.Integer(), nullable=True),
        sa.Column('climate_zone', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    op.create_index('idx_cities_name', 'cities', ['name'])

    # Create consumption_data table
    op.create_table(
        'consumption_data',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('city_id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('consumption', sa.Float(), nullable=False),
        sa.Column('temperature', sa.Float(), nullable=True),
        sa.Column('humidity', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['city_id'], ['cities.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_consumption_timestamp', 'consumption_data', ['timestamp'])
    op.create_index('idx_consumption_city', 'consumption_data', ['city_id'])

    # Create forecasts table
    op.create_table(
        'forecasts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('city_id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('predicted_consumption', sa.Float(), nullable=False),
        sa.Column('confidence_level', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['city_id'], ['cities.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_forecasts_timestamp', 'forecasts', ['timestamp'])
    op.create_index('idx_forecasts_city', 'forecasts', ['city_id'])

def downgrade():
    op.drop_table('forecasts')
    op.drop_table('consumption_data')
    op.drop_table('cities')
