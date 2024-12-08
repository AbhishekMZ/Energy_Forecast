"""Security tests for the Energy Forecast Platform."""

import pytest
import asyncio
import aiohttp
import jwt
import time
from datetime import datetime, timedelta
import json
from typing import Dict
import redis
from energy_forecast.core.security.manager import SecurityManager
from energy_forecast.core.security.threat_detection import ThreatDetectionSystem

class TestSecurity:
    """Security test suite."""

    @pytest.fixture
    def security_manager(self):
        redis_client = redis.Redis()
        return SecurityManager(
            redis_client=redis_client,
            secret_key="test_secret",
            encryption_key="test_encryption_key",
            jwt_secret="test_jwt_secret"
        )

    @pytest.fixture
    def threat_detection(self):
        redis_client = redis.Redis()
        config = {
            "max_auth_attempts": 5,
            "ddos_threshold_1m": 100,
            "ddos_threshold_5m": 300,
            "ddos_threshold_1h": 1000,
            "max_data_transfer": 1000000,
            "anomaly_threshold": -0.5,
            "bot_threshold": 0.8
        }
        return ThreatDetectionSystem(redis_client, config)

    async def test_brute_force_detection(self, threat_detection):
        """Test brute force attack detection."""
        request_data = {
            "source_ip": "192.168.1.1",
            "endpoint": "/api/v1/auth",
            "method": "POST"
        }

        # Simulate multiple failed attempts
        threats = []
        for _ in range(6):
            threat = await threat_detection._detect_brute_force(
                request_data,
                "test_request_id"
            )
            if threat:
                threats.append(threat)

        assert len(threats) > 0
        assert threats[-1].threat_type.value == "brute_force"
        assert threats[-1].confidence > 0.8

    async def test_ddos_detection(self, threat_detection):
        """Test DDoS attack detection."""
        request_data = {
            "source_ip": "192.168.1.2",
            "endpoint": "/api/v1/forecast",
            "method": "GET"
        }

        # Simulate high request rate
        threats = []
        for _ in range(101):
            threat = await threat_detection._detect_ddos(
                request_data,
                "test_request_id"
            )
            if threat:
                threats.append(threat)

        assert len(threats) > 0
        assert threats[-1].threat_type.value == "ddos"
        assert threats[-1].confidence > 0.7

    async def test_injection_detection(self, threat_detection):
        """Test SQL injection detection."""
        request_data = {
            "source_ip": "192.168.1.3",
            "endpoint": "/api/v1/data",
            "method": "POST",
            "payload": {
                "query": "SELECT * FROM users; DROP TABLE users;"
            }
        }

        threat = await threat_detection._detect_injection(
            request_data,
            "test_request_id"
        )

        assert threat is not None
        assert threat.threat_type.value == "injection"
        assert threat.confidence > 0.9

    async def test_data_exfiltration(self, threat_detection):
        """Test data exfiltration detection."""
        # Generate large payload
        large_payload = {"data": "x" * 1000000}
        
        request_data = {
            "source_ip": "192.168.1.4",
            "endpoint": "/api/v1/data",
            "method": "GET",
            "payload": large_payload
        }

        threat = await threat_detection._detect_data_exfiltration(
            request_data,
            "test_request_id"
        )

        assert threat is not None
        assert threat.threat_type.value == "data_exfiltration"
        assert threat.confidence > 0.7

    async def test_bot_detection(self, threat_detection):
        """Test bot behavior detection."""
        request_data = {
            "source_ip": "192.168.1.5",
            "endpoint": "/api/v1/forecast",
            "method": "GET",
            "headers": {
                "user-agent": "suspicious-bot/1.0"
            }
        }

        # Simulate rapid requests
        threats = []
        for _ in range(50):
            threat = await threat_detection._detect_bot_behavior(
                request_data,
                "test_request_id"
            )
            if threat:
                threats.append(threat)
            await asyncio.sleep(0.01)

        assert len(threats) > 0
        assert threats[-1].threat_type.value == "bot"

    def test_token_validation(self, security_manager):
        """Test JWT token validation."""
        # Generate token
        token = security_manager.generate_jwt_token(
            user_id="test_user",
            expiry_hours=1
        )

        # Validate token
        is_valid, payload = security_manager.validate_jwt_token(token)
        assert is_valid
        assert payload["user_id"] == "test_user"

        # Test expired token
        expired_token = security_manager.generate_jwt_token(
            user_id="test_user",
            expiry_hours=-1
        )
        is_valid, error = security_manager.validate_jwt_token(expired_token)
        assert not is_valid
        assert "expired" in error["error"].lower()

    def test_encryption(self, security_manager):
        """Test data encryption/decryption."""
        sensitive_data = "sensitive_information"
        
        # Encrypt data
        encrypted = security_manager.encrypt_sensitive_data(sensitive_data)
        assert encrypted != sensitive_data

        # Decrypt data
        decrypted = security_manager.decrypt_sensitive_data(encrypted)
        assert decrypted == sensitive_data

    async def test_rate_limiting(self, security_manager):
        """Test rate limiting functionality."""
        source_ip = "192.168.1.6"
        
        # Make requests up to limit
        for _ in range(100):
            is_allowed = await security_manager._check_rate_limits(source_ip)
            assert is_allowed

        # Verify rate limit exceeded
        is_allowed = await security_manager._check_rate_limits(source_ip)
        assert not is_allowed

    def test_input_validation(self, security_manager):
        """Test input validation."""
        # Test SQL injection
        payload = {
            "query": "SELECT * FROM users WHERE id = 1; DROP TABLE users;"
        }
        assert not security_manager._validate_payload(payload)

        # Test XSS
        payload = {
            "content": "<script>alert('xss')</script>"
        }
        assert not security_manager._validate_payload(payload)

        # Test valid payload
        payload = {
            "city": "Mumbai",
            "date": "2024-12-08"
        }
        assert security_manager._validate_payload(payload)

    async def test_threat_analysis(self, threat_detection):
        """Test threat analysis functionality."""
        # Generate some threat events
        request_data = {
            "source_ip": "192.168.1.7",
            "endpoint": "/api/v1/auth",
            "method": "POST"
        }

        # Simulate various threats
        for _ in range(10):
            await threat_detection.analyze_request(
                request_data,
                "test_request_id"
            )

        # Analyze threats
        analysis = await threat_detection.analyze_threats(time_window=3600)
        assert len(analysis) > 0

    def test_ip_validation(self, security_manager):
        """Test IP address validation."""
        # Test valid public IP
        assert security_manager._validate_ip("8.8.8.8")

        # Test private IP
        assert not security_manager._validate_ip("192.168.1.1")

        # Test invalid IP
        assert not security_manager._validate_ip("invalid_ip")

    async def test_concurrent_security(self, security_manager):
        """Test security under concurrent load."""
        async def make_request():
            return await security_manager.validate_request(
                request_id="test_request",
                source_ip="8.8.8.8",
                payload={"test": "data"},
                headers={"user-agent": "test"}
            )

        # Make concurrent requests
        tasks = [make_request() for _ in range(100)]
        results = await asyncio.gather(*tasks)

        # Verify results
        assert all(result[0] for result in results)

    def test_security_headers(self, security_manager):
        """Test security headers validation."""
        # Test valid headers
        headers = {
            "user-agent": "test-agent",
            "host": "api.energyforecast.com",
            "content-type": "application/json"
        }
        assert security_manager._validate_headers(headers)

        # Test missing required headers
        headers = {
            "content-type": "application/json"
        }
        assert not security_manager._validate_headers(headers)

        # Test invalid content type
        headers = {
            "user-agent": "test-agent",
            "host": "api.energyforecast.com",
            "content-type": "text/plain"
        }
        assert not security_manager._validate_headers(headers)

    @pytest.mark.asyncio
    async def test_security_event_logging(self, threat_detection):
        """Test security event logging."""
        request_data = {
            "source_ip": "192.168.1.8",
            "endpoint": "/api/v1/data",
            "method": "POST",
            "payload": {
                "query": "SELECT * FROM users; DROP TABLE users;"
            }
        }

        # Generate threat
        threat = await threat_detection._detect_injection(
            request_data,
            "test_request_id"
        )

        # Handle threat
        await threat_detection._handle_threat(threat)

        # Verify logged event
        events = await threat_detection.analyze_threats(time_window=60)
        assert "injection" in str(events)

    def test_sensitive_data_detection(self, threat_detection):
        """Test sensitive data detection."""
        # Test credit card number
        payload = {
            "card": "4111-1111-1111-1111"
        }
        assert threat_detection._contains_sensitive_data(payload)

        # Test email
        payload = {
            "email": "test@example.com"
        }
        assert threat_detection._contains_sensitive_data(payload)

        # Test SSN
        payload = {
            "ssn": "123-45-6789"
        }
        assert threat_detection._contains_sensitive_data(payload)

        # Test API key
        payload = {
            "api_key": "a" * 32
        }
        assert threat_detection._contains_sensitive_data(payload)
