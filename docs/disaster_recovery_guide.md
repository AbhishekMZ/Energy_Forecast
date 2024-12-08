# Disaster Recovery Guide

## Overview

This guide outlines the disaster recovery (DR) procedures for the Energy Forecast Platform, ensuring business continuity in the event of system failures, data loss, or catastrophic events.

## Recovery Time Objectives (RTO) and Recovery Point Objectives (RPO)

| Service Component | RTO | RPO |
|------------------|-----|-----|
| API Services | 15 minutes | < 1 minute |
| ML Models | 30 minutes | < 5 minutes |
| Databases | 1 hour | < 1 minute |
| Cache | 5 minutes | Best effort |
| Model Training | 4 hours | Last successful training |

## Backup Procedures

### 1. Database Backups

```bash
#!/bin/bash
# backup_databases.sh

# PostgreSQL backup
pg_dump -h $DB_HOST -U $DB_USER -d energy_forecast | \
  gzip > /backups/db/energy_forecast_$(date +%Y%m%d_%H%M%S).sql.gz

# TimescaleDB backup
pg_dump -h $TSDB_HOST -U $TSDB_USER -d timeseries | \
  gzip > /backups/tsdb/timeseries_$(date +%Y%m%d_%H%M%S).sql.gz

# Upload to S3
aws s3 sync /backups s3://$BACKUP_BUCKET/databases/
```

### 2. Model Artifacts

```python
# backup_models.py
def backup_models():
    """Backup ML model artifacts."""
    backup_paths = {
        'models': '/models',
        'configs': '/configs',
        'metadata': '/metadata'
    }
    
    for name, path in backup_paths.items():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f'{name}_{timestamp}.tar.gz'
        
        # Create compressed archive
        with tarfile.open(backup_name, 'w:gz') as tar:
            tar.add(path)
        
        # Upload to S3
        s3_client.upload_file(
            backup_name,
            BACKUP_BUCKET,
            f'models/{backup_name}'
        )
```

### 3. Configuration Backups

```yaml
# backup-config.yaml
schedule:
  database:
    full:
      frequency: "0 0 * * *"  # Daily at midnight
      retention: "30d"
    incremental:
      frequency: "0 */6 * * *"  # Every 6 hours
      retention: "7d"
  
  models:
    frequency: "0 */12 * * *"  # Every 12 hours
    retention: "14d"
  
  configs:
    frequency: "0 0 * * *"  # Daily
    retention: "90d"
```

## Disaster Scenarios and Recovery Procedures

### 1. Database Failure

```bash
# database_recovery.sh
#!/bin/bash

# 1. Stop affected services
kubectl scale deployment api-service --replicas=0

# 2. Restore database from latest backup
latest_backup=$(aws s3 ls s3://$BACKUP_BUCKET/databases/latest/ | sort | tail -n 1)
aws s3 cp s3://$BACKUP_BUCKET/databases/latest/$latest_backup .
gunzip < $latest_backup | psql -h $DB_HOST -U $DB_USER -d energy_forecast

# 3. Apply transaction logs
psql -h $DB_HOST -U $DB_USER -d energy_forecast -f transaction_logs.sql

# 4. Verify data integrity
python verify_data_integrity.py

# 5. Restart services
kubectl scale deployment api-service --replicas=3
```

### 2. Model Service Failure

```python
# model_recovery.py
def recover_model_service():
    """Recover ML model service."""
    try:
        # 1. Stop prediction service
        scale_deployment('ml-service', 0)
        
        # 2. Restore model artifacts
        restore_model_artifacts()
        
        # 3. Verify model integrity
        verify_model_integrity()
        
        # 4. Restart service
        scale_deployment('ml-service', 3)
        
        # 5. Monitor predictions
        monitor_predictions(duration='1h')
        
    except Exception as e:
        trigger_alert(f"Model recovery failed: {str(e)}")
        rollback_to_previous_version()
```

### 3. Complete System Recovery

```python
# system_recovery.py
def perform_system_recovery():
    """Perform complete system recovery."""
    recovery_steps = [
        ('infrastructure', recover_infrastructure),
        ('databases', recover_databases),
        ('cache', recover_cache),
        ('models', recover_models),
        ('services', recover_services)
    ]
    
    for step_name, step_func in recovery_steps:
        try:
            logger.info(f"Starting recovery step: {step_name}")
            step_func()
            logger.info(f"Completed recovery step: {step_name}")
        except Exception as e:
            logger.error(f"Recovery step failed: {step_name}")
            trigger_alert(f"Recovery failed at {step_name}: {str(e)}")
            return False
    
    return True
```

## Failover Procedures

### 1. Region Failover

```python
# region_failover.py
def initiate_region_failover():
    """Initiate failover to backup region."""
    try:
        # 1. Verify backup region health
        verify_backup_region()
        
        # 2. Switch DNS records
        update_dns_records()
        
        # 3. Promote read replicas
        promote_read_replicas()
        
        # 4. Scale up services
        scale_services_in_backup_region()
        
        # 5. Verify system health
        verify_system_health()
        
    except Exception as e:
        trigger_alert(f"Region failover failed: {str(e)}")
        rollback_failover()
```

### 2. Database Failover

```python
# database_failover.py
def perform_database_failover():
    """Perform database failover to standby."""
    try:
        # 1. Verify standby is up to date
        verify_standby_sync()
        
        # 2. Promote standby to primary
        promote_standby()
        
        # 3. Update connection strings
        update_connection_strings()
        
        # 4. Verify replication
        verify_replication()
        
    except Exception as e:
        trigger_alert(f"Database failover failed: {str(e)}")
        initiate_manual_recovery()
```

## Recovery Testing

### 1. Regular Testing Schedule

```yaml
# recovery-testing.yaml
schedule:
  database_recovery:
    frequency: monthly
    type: full
    notification: true
  
  model_recovery:
    frequency: weekly
    type: partial
    notification: true
  
  region_failover:
    frequency: quarterly
    type: simulation
    notification: true
```

### 2. Testing Procedures

```python
# recovery_testing.py
def conduct_recovery_test():
    """Conduct recovery test."""
    test_scenarios = [
        ('database_failure', test_database_recovery),
        ('model_failure', test_model_recovery),
        ('region_failure', test_region_failover),
        ('complete_failure', test_complete_recovery)
    ]
    
    results = {}
    for scenario, test_func in test_scenarios:
        try:
            start_time = time.time()
            success = test_func()
            duration = time.time() - start_time
            
            results[scenario] = {
                'success': success,
                'duration': duration,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            results[scenario] = {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    generate_test_report(results)
```

## Communication Plan

### 1. Notification Templates

```python
# notifications.py
NOTIFICATION_TEMPLATES = {
    'outage_detected': """
        ALERT: System outage detected
        Time: {timestamp}
        Component: {component}
        Impact: {impact}
        Recovery ETA: {eta}
        Actions: {actions}
    """,
    
    'recovery_started': """
        UPDATE: Recovery procedure initiated
        Time: {timestamp}
        Component: {component}
        Estimated Duration: {duration}
        Current Status: {status}
    """,
    
    'recovery_completed': """
        UPDATE: Recovery completed
        Time: {timestamp}
        Component: {component}
        Duration: {duration}
        Status: {status}
        Next Steps: {next_steps}
    """
}
```

### 2. Contact Matrix

```yaml
# contacts.yaml
teams:
  infrastructure:
    primary: "infra-lead@company.com"
    secondary: "infra-backup@company.com"
    phone: "+1234567890"
  
  database:
    primary: "db-lead@company.com"
    secondary: "db-backup@company.com"
    phone: "+1234567891"
  
  ml:
    primary: "ml-lead@company.com"
    secondary: "ml-backup@company.com"
    phone: "+1234567892"
```

## Recovery Metrics

### 1. Key Metrics

```python
# recovery_metrics.py
RECOVERY_METRICS = {
    'time_to_detect': {
        'threshold': '5m',
        'alert': True
    },
    'time_to_respond': {
        'threshold': '15m',
        'alert': True
    },
    'time_to_recover': {
        'threshold': {
            'critical': '1h',
            'high': '4h',
            'medium': '12h'
        },
        'alert': True
    },
    'data_loss': {
        'threshold': '0',
        'alert': True
    }
}
```

### 2. Monitoring

```python
# recovery_monitoring.py
def monitor_recovery():
    """Monitor recovery progress."""
    metrics = {
        'service_health': check_service_health(),
        'data_integrity': verify_data_integrity(),
        'model_performance': check_model_performance(),
        'api_latency': measure_api_latency(),
        'error_rates': calculate_error_rates()
    }
    
    # Log metrics
    log_recovery_metrics(metrics)
    
    # Check against thresholds
    alerts = check_metric_thresholds(metrics)
    
    # Send alerts if needed
    if alerts:
        send_metric_alerts(alerts)
```

## Additional Resources

- [Infrastructure Guide](./infrastructure_guide.md)
- [Monitoring Guide](./monitoring_guide.md)
- [Security Guide](./security_guide.md)
- [Database Schema](./database_schema.md)
- [Model Training Guide](./model_training_guide.md)
