"""create power suppliers table

Revision ID: 001
Revises: 
Create Date: 2023-11-20

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    supplier_type = sa.Enum('solar', 'wind', 'hydro', 'thermal', 'nuclear', 'other',
                           name='supplier_type')
    
    op.create_table(
        'power_suppliers',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(), nullable=False),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('hashed_password', sa.String(), nullable=False),
        sa.Column('full_name', sa.String(), nullable=False),
        sa.Column('company_name', sa.String(), nullable=False),
        sa.Column('supplier_type', supplier_type, nullable=False),
        sa.Column('license_number', sa.String(), nullable=False),
        sa.Column('capacity_mw', sa.Float(), nullable=False),
        sa.Column('location', sa.String(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('is_verified', sa.Boolean(), nullable=False, default=False),
        sa.Column('role', sa.String(), nullable=False, default='supplier'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('last_updated', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('username'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('license_number')
    )
    
    # Create indexes for frequently accessed columns
    op.create_index(op.f('ix_power_suppliers_username'), 'power_suppliers', ['username'])
    op.create_index(op.f('ix_power_suppliers_email'), 'power_suppliers', ['email'])
    op.create_index(op.f('ix_power_suppliers_license_number'), 'power_suppliers', ['license_number'])

def downgrade():
    op.drop_table('power_suppliers')
    op.execute('DROP TYPE supplier_type')
