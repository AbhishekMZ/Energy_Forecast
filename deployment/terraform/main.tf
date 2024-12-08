terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
  
  backend "s3" {
    bucket = "energy-forecast-terraform-state"
    key    = "state/terraform.tfstate"
    region = "us-east-1"
    
    dynamodb_table = "terraform-state-lock"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment = var.environment
      Project     = "energy-forecast"
      ManagedBy   = "terraform"
    }
  }
}

# VPC Configuration
module "vpc" {
  source = "./modules/vpc"
  
  environment         = var.environment
  vpc_cidr           = var.vpc_cidr
  availability_zones = var.availability_zones
  private_subnets    = var.private_subnets
  public_subnets     = var.public_subnets
}

# ECS Cluster
module "ecs" {
  source = "./modules/ecs"
  
  environment     = var.environment
  vpc_id         = module.vpc.vpc_id
  private_subnets = module.vpc.private_subnets
  
  app_image      = var.app_image
  app_count      = var.app_count
  fargate_cpu    = var.fargate_cpu
  fargate_memory = var.fargate_memory
  
  depends_on = [module.vpc]
}

# RDS Database
module "rds" {
  source = "./modules/rds"
  
  environment     = var.environment
  vpc_id         = module.vpc.vpc_id
  private_subnets = module.vpc.private_subnets
  
  db_name     = var.db_name
  db_username = var.db_username
  db_password = var.db_password
  
  depends_on = [module.vpc]
}

# Redis Cluster
module "redis" {
  source = "./modules/redis"
  
  environment     = var.environment
  vpc_id         = module.vpc.vpc_id
  private_subnets = module.vpc.private_subnets
  
  redis_node_type = var.redis_node_type
  redis_nodes     = var.redis_nodes
  
  depends_on = [module.vpc]
}

# Application Load Balancer
module "alb" {
  source = "./modules/alb"
  
  environment    = var.environment
  vpc_id        = module.vpc.vpc_id
  public_subnets = module.vpc.public_subnets
  
  certificate_arn = var.certificate_arn
  
  depends_on = [module.vpc]
}

# Security Groups
module "security_groups" {
  source = "./modules/security"
  
  environment = var.environment
  vpc_id     = module.vpc.vpc_id
  
  depends_on = [module.vpc]
}

# CloudWatch Monitoring
module "monitoring" {
  source = "./modules/monitoring"
  
  environment = var.environment
  
  alarm_email = var.alarm_email
  
  depends_on = [module.ecs, module.rds, module.redis]
}

# S3 Buckets
module "s3" {
  source = "./modules/s3"
  
  environment = var.environment
  
  model_bucket_name = var.model_bucket_name
  logs_bucket_name  = var.logs_bucket_name
}

# Route53 DNS
module "dns" {
  source = "./modules/dns"
  
  domain_name = var.domain_name
  alb_dns_name = module.alb.dns_name
  alb_zone_id  = module.alb.zone_id
  
  depends_on = [module.alb]
}

# WAF Configuration
module "waf" {
  source = "./modules/waf"
  
  environment = var.environment
  alb_arn    = module.alb.arn
  
  depends_on = [module.alb]
}

# Backup Configuration
module "backup" {
  source = "./modules/backup"
  
  environment = var.environment
  
  rds_arn     = module.rds.arn
  backup_retention = var.backup_retention
  
  depends_on = [module.rds]
}

# Outputs
output "alb_dns_name" {
  value = module.alb.dns_name
}

output "ecs_cluster_name" {
  value = module.ecs.cluster_name
}

output "rds_endpoint" {
  value = module.rds.endpoint
}

output "redis_endpoint" {
  value = module.redis.endpoint
}

output "model_bucket_name" {
  value = module.s3.model_bucket_name
}

output "logs_bucket_name" {
  value = module.s3.logs_bucket_name
}
