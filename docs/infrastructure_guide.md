# Infrastructure Guide

## Overview

This guide details the cloud infrastructure setup and management for the Energy Forecast Platform, including AWS resources, Kubernetes clusters, and monitoring systems.

## Infrastructure as Code

### 1. Terraform Configuration

```hcl
# main.tf
provider "aws" {
  region = var.aws_region
}

# VPC Configuration
module "vpc" {
  source = "./modules/vpc"
  
  vpc_cidr = "10.0.0.0/16"
  azs = ["ap-south-1a", "ap-south-1b", "ap-south-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway = true
  single_nat_gateway = false
  
  tags = {
    Environment = var.environment
    Project = "energy-forecast"
  }
}

# EKS Cluster
module "eks" {
  source = "./modules/eks"
  
  cluster_name = "energy-forecast-${var.environment}"
  cluster_version = "1.24"
  
  vpc_id = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  node_groups = {
    general = {
      desired_capacity = 3
      max_capacity = 5
      min_capacity = 2
      instance_types = ["t3.large"]
    }
    compute = {
      desired_capacity = 2
      max_capacity = 4
      min_capacity = 1
      instance_types = ["c5.2xlarge"]
      labels = {
        workload = "compute"
      }
    }
  }
}

# RDS Instance
module "rds" {
  source = "./modules/rds"
  
  identifier = "energy-forecast-${var.environment}"
  engine = "postgres"
  engine_version = "13.7"
  instance_class = "db.t3.large"
  allocated_storage = 100
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  subnet_ids = module.vpc.private_subnets
  
  database_name = "energy_forecast"
  port = "5432"
  
  backup_retention_period = 7
  backup_window = "03:00-04:00"
  maintenance_window = "Mon:04:00-Mon:05:00"
  
  multi_az = true
  
  tags = {
    Environment = var.environment
  }
}

# ElastiCache Redis
module "redis" {
  source = "./modules/elasticache"
  
  cluster_id = "energy-forecast-${var.environment}"
  engine = "redis"
  engine_version = "6.x"
  node_type = "cache.t3.medium"
  num_cache_nodes = 2
  
  subnet_group_name = aws_elasticache_subnet_group.redis.name
  security_group_ids = [aws_security_group.redis.id]
  
  automatic_failover_enabled = true
  multi_az_enabled = true
  
  tags = {
    Environment = var.environment
  }
}
```

### 2. Kubernetes Manifests

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: energy-forecast-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: energy-forecast-api
  template:
    metadata:
      labels:
        app: energy-forecast-api
    spec:
      containers:
      - name: api
        image: energy-forecast-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "1000m"
            memory: "1Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: url

---
# kubernetes/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: energy-forecast-api
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: energy-forecast-api

---
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: energy-forecast-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: energy-forecast-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Monitoring Setup

### 1. Prometheus Configuration

```yaml
# prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert.rules"

alerting:
  alertmanagers:
  - static_configs:
    - targets:
      - "alertmanager:9093"

scrape_configs:
  - job_name: 'kubernetes-apiservers'
    kubernetes_sd_configs:
    - role: endpoints
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token

  - job_name: 'kubernetes-nodes'
    kubernetes_sd_configs:
    - role: node
    relabel_configs:
    - action: labelmap
      regex: __meta_kubernetes_node_label_(.+)

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
    - role: pod
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
      action: keep
      regex: true
```

### 2. Grafana Dashboards

```json
{
  "dashboard": {
    "id": null,
    "title": "Energy Forecast Platform",
    "tags": ["kubernetes", "energy-forecast"],
    "timezone": "browser",
    "panels": [
      {
        "title": "API Request Rate",
        "type": "graph",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{app=\"energy-forecast-api\"}[5m])) by (status)",
            "legendFormat": "{{status}}"
          }
        ]
      },
      {
        "title": "Model Prediction Latency",
        "type": "graph",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(prediction_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "p95"
          }
        ]
      }
    ]
  }
}
```

## Security Configuration

### 1. Network Security

```hcl
# security_groups.tf
resource "aws_security_group" "eks_cluster" {
  name_prefix = "eks-cluster-"
  vpc_id = module.vpc.vpc_id

  ingress {
    from_port = 443
    to_port = 443
    protocol = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port = 0
    to_port = 0
    protocol = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "rds" {
  name_prefix = "rds-"
  vpc_id = module.vpc.vpc_id

  ingress {
    from_port = 5432
    to_port = 5432
    protocol = "tcp"
    security_groups = [aws_security_group.eks_cluster.id]
  }
}

resource "aws_security_group" "redis" {
  name_prefix = "redis-"
  vpc_id = module.vpc.vpc_id

  ingress {
    from_port = 6379
    to_port = 6379
    protocol = "tcp"
    security_groups = [aws_security_group.eks_cluster.id]
  }
}
```

### 2. IAM Configuration

```hcl
# iam.tf
resource "aws_iam_role" "eks_cluster" {
  name = "eks-cluster-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role = aws_iam_role.eks_cluster.name
}

resource "aws_iam_role" "eks_node_group" {
  name = "eks-node-group-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}
```

## Backup Configuration

### 1. Database Backups

```hcl
# rds_backup.tf
resource "aws_db_instance" "energy_forecast" {
  # ... other configuration ...

  backup_retention_period = 7
  backup_window = "03:00-04:00"
  maintenance_window = "Mon:04:00-Mon:05:00"

  copy_tags_to_snapshot = true
  deletion_protection = true

  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
}

resource "aws_db_snapshot" "daily_backup" {
  db_instance_identifier = aws_db_instance.energy_forecast.id
  db_snapshot_identifier = "energy-forecast-backup-${formatdate("YYYY-MM-DD", timestamp())}"
}
```

### 2. Application Backups

```yaml
# kubernetes/backup.yaml
apiVersion: batch/v1beta1
kind: CronJob
metadata:
  name: application-backup
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: energy-forecast-backup:latest
            command:
            - /backup.sh
            env:
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: aws-credentials
                  key: access_key
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: aws-credentials
                  key: secret_key
            - name: BACKUP_BUCKET
              value: "energy-forecast-backups"
          restartPolicy: OnFailure
```

## Scaling Configuration

### 1. Horizontal Pod Autoscaling

```yaml
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: energy-forecast-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: energy-forecast-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
    scaleUp:
      stabilizationWindowSeconds: 60
```

### 2. Vertical Pod Autoscaling

```yaml
# kubernetes/vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: energy-forecast-api
spec:
  targetRef:
    apiVersion: "apps/v1"
    kind: Deployment
    name: energy-forecast-api
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: '*'
      minAllowed:
        cpu: "100m"
        memory: "256Mi"
      maxAllowed:
        cpu: "2"
        memory: "4Gi"
```

## Disaster Recovery

### 1. Backup Strategy

```bash
#!/bin/bash
# backup.sh

# Database backup
pg_dump -h $DB_HOST -U $DB_USER -d energy_forecast | \
  gzip > backup_$(date +%Y%m%d).sql.gz

# Upload to S3
aws s3 cp backup_$(date +%Y%m%d).sql.gz \
  s3://$BACKUP_BUCKET/database/

# Model backup
aws s3 sync /models \
  s3://$BACKUP_BUCKET/models/
```

### 2. Recovery Plan

```bash
#!/bin/bash
# restore.sh

# Database restore
aws s3 cp s3://$BACKUP_BUCKET/database/backup_$DATE.sql.gz .
gunzip backup_$DATE.sql.gz
psql -h $DB_HOST -U $DB_USER -d energy_forecast < backup_$DATE.sql

# Model restore
aws s3 sync s3://$BACKUP_BUCKET/models/ /models/
```

## Additional Resources

- [Deployment Guide](./deployment_guide.md)
- [Security Guide](./security_guide.md)
- [Monitoring Guide](./monitoring_guide.md)
- [Database Guide](./database_schema.md)
