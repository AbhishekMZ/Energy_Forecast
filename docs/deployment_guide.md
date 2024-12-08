# Comprehensive Deployment Guide

## Overview

This guide provides comprehensive deployment instructions for the Energy Forecast Platform across different environments, including local development, staging, and production.

## Infrastructure Architecture

### Production Environment

```mermaid
graph TD
    Client[Client Applications] --> LB[Load Balancer]
    LB --> API[API Service]
    API --> Cache[Redis Cache]
    API --> DB[(Database)]
    API --> ML[ML Service]
    ML --> Models[(Model Storage)]
```

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.20+
- AWS CLI 2.0+
- Terraform 1.0+

## Deployment Process

### 1. Infrastructure Setup

```hcl
# main.tf
provider "aws" {
  region = "ap-south-1"
}

module "vpc" {
  source = "./modules/vpc"
  cidr_block = "10.0.0.0/16"
  availability_zones = ["ap-south-1a", "ap-south-1b"]
}

module "ecs" {
  source = "./modules/ecs"
  vpc_id = module.vpc.vpc_id
  subnets = module.vpc.private_subnets
}

module "rds" {
  source = "./modules/rds"
  vpc_id = module.vpc.vpc_id
  subnets = module.vpc.database_subnets
}

module "elasticache" {
  source = "./modules/elasticache"
  vpc_id = module.vpc.vpc_id
  subnets = module.vpc.elasticache_subnets
}
```

### 2. Container Build

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
```

### 3. CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ap-south-1
      
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1
      
      - name: Build and push
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: energy-forecast
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
      
      - name: Deploy to ECS
        run: |
          aws ecs update-service --cluster energy-forecast \
            --service api --force-new-deployment
```

## Environment Configuration

### 1. Environment Variables

```bash
# .env.production
FLASK_ENV=production
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379/0
AWS_REGION=ap-south-1
MODEL_BUCKET=energy-forecast-models
LOG_LEVEL=INFO
```

### 2. Secrets Management

```yaml
# AWS Secrets Manager
{
  "DATABASE_URL": "postgresql://user:pass@host:5432/db",
  "REDIS_URL": "redis://host:6379/0",
  "API_KEY": "your-api-key",
  "JWT_SECRET": "your-jwt-secret"
}
```

## Monitoring Setup

### 1. CloudWatch Configuration

```yaml
# cloudwatch.yml
metrics:
  namespace: EnergyForecast
  dimensions:
    - Service
    - Environment
    - Region

alarms:
  - name: HighCPUUtilization
    metric: CPUUtilization
    threshold: 80
    period: 300
    evaluation_periods: 2
    comparison_operator: GreaterThanThreshold
    
  - name: HighMemoryUtilization
    metric: MemoryUtilization
    threshold: 85
    period: 300
    evaluation_periods: 2
    comparison_operator: GreaterThanThreshold
```

### 2. Logging Configuration

```python
# logging_config.py
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            'format': '%(asctime)s %(levelname)s %(name)s %(message)s',
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter'
        }
    },
    'handlers': {
        'cloudwatch': {
            'class': 'watchtower.CloudWatchLogHandler',
            'log_group': 'energy-forecast',
            'stream_name': 'production',
            'formatter': 'json'
        }
    },
    'loggers': {
        '': {
            'level': 'INFO',
            'handlers': ['cloudwatch']
        }
    }
}
```

## Scaling Configuration

### 1. Auto Scaling

```yaml
# autoscaling.yml
auto_scaling:
  target_tracking:
    target_value: 70
    scale_in_cooldown: 300
    scale_out_cooldown: 300
    
  step_scaling:
    up:
      - threshold: 70
        scaling_adjustment: 1
      - threshold: 85
        scaling_adjustment: 2
    down:
      - threshold: 40
        scaling_adjustment: -1
```

### 2. Resource Allocation

```yaml
# task-definition.json
{
  "family": "energy-forecast",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "energy-forecast:latest",
      "cpu": 1024,
      "memory": 2048,
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "hostPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "FLASK_ENV",
          "value": "production"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "energy-forecast",
          "awslogs-region": "ap-south-1",
          "awslogs-stream-prefix": "api"
        }
      }
    }
  ]
}
```

## Backup Strategy

### 1. Database Backups

```yaml
# rds-backup.yml
backup_configuration:
  automated_backups:
    retention_period: 7
    backup_window: "00:00-01:00"
    
  snapshot_schedule:
    frequency: daily
    retention_period: 30
    copy_to_region: ap-southeast-1
```

### 2. S3 Lifecycle Rules

```yaml
# s3-lifecycle.yml
lifecycle_rules:
  - id: "model-artifacts"
    prefix: "models/"
    transitions:
      - days: 30
        storage_class: STANDARD_IA
      - days: 90
        storage_class: GLACIER
    
  - id: "training-data"
    prefix: "data/"
    transitions:
      - days: 60
        storage_class: GLACIER
```

## Disaster Recovery

### 1. Recovery Point Objective (RPO)

- Database: 5 minutes
- Cache: Real-time replication
- Storage: Cross-region replication

### 2. Recovery Time Objective (RTO)

- Critical services: < 1 hour
- Non-critical services: < 4 hours

### 3. Recovery Procedures

```bash
# recovery-steps.sh
#!/bin/bash

# 1. Restore database
aws rds restore-db-instance-from-snapshot \
  --db-instance-identifier energy-forecast-prod \
  --db-snapshot-identifier latest-backup

# 2. Update DNS
aws route53 change-resource-record-sets \
  --hosted-zone-id ZONE_ID \
  --change-batch file://dns-update.json

# 3. Scale up services
aws ecs update-service \
  --cluster energy-forecast \
  --service api \
  --desired-count 3
```

## Security Measures

### 1. Network Security

```hcl
# security-groups.tf
resource "aws_security_group" "api" {
  name = "api-sg"
  vpc_id = var.vpc_id

  ingress {
    from_port = 8000
    to_port = 8000
    protocol = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }
}

resource "aws_security_group" "database" {
  name = "database-sg"
  vpc_id = var.vpc_id

  ingress {
    from_port = 5432
    to_port = 5432
    protocol = "tcp"
    security_groups = [aws_security_group.api.id]
  }
}
```

### 2. IAM Policies

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::energy-forecast-*/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": "arn:aws:secretsmanager:*:*:secret:energy-forecast-*"
    }
  ]
}
```

## Performance Optimization

### 1. Caching Strategy

```python
# cache_config.py
CACHING_CONFIG = {
    'default_timeout': 3600,
    'key_prefix': 'energy_forecast',
    'redis_url': 'redis://localhost:6379/0',
    'regions': {
        'forecast': {
            'timeout': 1800,
            'key_pattern': 'forecast:{city}:{date}'
        },
        'model': {
            'timeout': 3600,
            'key_pattern': 'model:{version}:{feature}'
        }
    }
}
```

### 2. Database Optimization

```sql
-- database_indexes.sql
CREATE INDEX idx_consumption_city_date ON consumption(city, date);
CREATE INDEX idx_weather_city_date ON weather(city, date);
CREATE INDEX idx_model_version_status ON models(version, status);
```

## Maintenance Procedures

### 1. Database Maintenance

```bash
# maintenance.sh
#!/bin/bash

# Vacuum database
psql $DATABASE_URL -c "VACUUM ANALYZE;"

# Reindex
psql $DATABASE_URL -c "REINDEX DATABASE energy_forecast;"

# Update statistics
psql $DATABASE_URL -c "ANALYZE VERBOSE;"
```

### 2. Cache Maintenance

```python
# cache_maintenance.py
def maintain_cache():
    """Perform cache maintenance tasks."""
    # Clear expired keys
    redis_client.execute_command('SCAN', 0, 'COUNT', 1000)
    
    # Compact database
    redis_client.execute_command('BGREWRITEAOF')
```

## Rollback Procedures

### 1. Application Rollback

```bash
# rollback.sh
#!/bin/bash

# Get previous task definition
PREVIOUS_TASK_DEF=$(aws ecs describe-task-definition \
  --task-definition energy-forecast:$PREVIOUS_VERSION)

# Update service
aws ecs update-service \
  --cluster energy-forecast \
  --service api \
  --task-definition $PREVIOUS_TASK_DEF
```

### 2. Database Rollback

```sql
-- rollback.sql
BEGIN;
  -- Restore from backup
  SELECT pg_restore('backup.sql');
  
  -- Verify data
  SELECT COUNT(*) FROM consumption;
  
COMMIT;
```

## Troubleshooting Guide

### Common Issues

1. High CPU Usage
```bash
# Check container metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --dimensions Name=ServiceName,Value=api \
  --start-time $(date -u +"%Y-%m-%dT%H:%M:%SZ" -d "1 hour ago") \
  --end-time $(date -u +"%Y-%m-%dT%H:%M:%SZ") \
  --period 300 \
  --statistics Average
```

2. Memory Leaks
```python
# memory_profile.py
import memory_profiler

@profile
def check_memory_usage():
    """Profile memory usage of key functions."""
    # Your code here
```

3. Slow Queries
```sql
-- slow_queries.sql
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;
```

## Deployment Checklist

### Pre-deployment
- [ ] Run test suite
- [ ] Check resource utilization
- [ ] Backup database
- [ ] Update documentation
- [ ] Review security settings

### Deployment
- [ ] Deploy infrastructure changes
- [ ] Update application code
- [ ] Run database migrations
- [ ] Update configuration
- [ ] Deploy monitoring updates

### Post-deployment
- [ ] Verify application health
- [ ] Check metrics and logs
- [ ] Run smoke tests
- [ ] Monitor error rates
- [ ] Update status page
