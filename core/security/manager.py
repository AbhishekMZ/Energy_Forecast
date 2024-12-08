"""Advanced security manager for the Energy Forecast Platform."""

import hashlib
import hmac
import base64
from cryptography.fernet import Fernet
from datetime import datetime, timedelta
import jwt
import redis
import logging
from typing import Dict, Optional, List, Tuple
import ipaddress
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityEvent:
    timestamp: datetime
    event_type: str
    threat_level: ThreatLevel
    details: Dict
    source_ip: str
    user_id: Optional[str]

class SecurityManager:
    def __init__(
        self,
        redis_client: redis.Redis,
        secret_key: str,
        encryption_key: str,
        jwt_secret: str
    ):
        self.redis = redis_client
        self.secret_key = secret_key
        self.fernet = Fernet(encryption_key)
        self.jwt_secret = jwt_secret
        
        # Initialize security components
        self._init_security_components()

    def _init_security_components(self):
        """Initialize security components and configurations."""
        # Threat patterns
        self.threat_patterns = {
            "sql_injection": re.compile(
                r"(\b(union|select|insert|delete|from|drop table|update|exec)\b)",
                re.IGNORECASE
            ),
            "xss": re.compile(
                r"(<script|javascript:|data:text/html|<img|onerror=|onload=)",
                re.IGNORECASE
            ),
            "path_traversal": re.compile(r"\.\.\/|\.\.\\"),
            "command_injection": re.compile(r"[;&|`]")
        }
        
        # IP blacklist/whitelist
        self.ip_blacklist = set()
        self.ip_whitelist = set()
        
        # Rate limiting windows
        self.rate_windows = {
            "1m": 60,
            "5m": 300,
            "1h": 3600,
            "24h": 86400
        }

    async def validate_request(
        self,
        request_id: str,
        source_ip: str,
        payload: Dict,
        headers: Dict
    ) -> Tuple[bool, Optional[str]]:
        """Comprehensive request validation."""
        try:
            # Basic checks
            if not all([
                self._validate_ip(source_ip),
                self._validate_payload(payload),
                self._validate_headers(headers)
            ]):
                return False, "Basic validation failed"

            # Advanced security checks
            security_checks = await self._perform_security_checks(
                request_id, source_ip, payload
            )
            if not security_checks[0]:
                return security_checks

            # Rate limiting
            if not await self._check_rate_limits(source_ip):
                return False, "Rate limit exceeded"

            return True, None

        except Exception as e:
            logger.error(f"Security validation error: {str(e)}")
            return False, "Security validation error"

    def _validate_ip(self, ip: str) -> bool:
        """Validate IP address."""
        try:
            ip_obj = ipaddress.ip_address(ip)
            
            # Check blacklist/whitelist
            if str(ip_obj) in self.ip_blacklist:
                return False
            if self.ip_whitelist and str(ip_obj) not in self.ip_whitelist:
                return False
                
            # Check for private IPs in public endpoints
            if ip_obj.is_private and not self._is_internal_request():
                return False
                
            return True
        except ValueError:
            return False

    def _validate_payload(self, payload: Dict) -> bool:
        """Validate request payload for security threats."""
        try:
            # Convert payload to string for pattern matching
            payload_str = str(payload)
            
            # Check for known threat patterns
            for pattern_name, pattern in self.threat_patterns.items():
                if pattern.search(payload_str):
                    self._log_security_event(
                        event_type=f"threat_detected_{pattern_name}",
                        threat_level=ThreatLevel.HIGH,
                        details={"payload": payload_str}
                    )
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Payload validation error: {str(e)}")
            return False

    def _validate_headers(self, headers: Dict) -> bool:
        """Validate request headers."""
        required_headers = {"user-agent", "host"}
        if not all(h in headers for h in required_headers):
            return False
            
        # Validate content type
        content_type = headers.get("content-type", "")
        if content_type and "application/json" not in content_type:
            return False
            
        return True

    async def _perform_security_checks(
        self,
        request_id: str,
        source_ip: str,
        payload: Dict
    ) -> Tuple[bool, Optional[str]]:
        """Perform advanced security checks."""
        # Check for automated threats
        if await self._detect_automated_threats(source_ip):
            return False, "Automated threat detected"
            
        # Check for anomalies
        if await self._detect_anomalies(payload):
            return False, "Anomalous behavior detected"
            
        # Validate request signature
        if not self._validate_signature(request_id, payload):
            return False, "Invalid request signature"
            
        return True, None

    async def _check_rate_limits(self, source_ip: str) -> bool:
        """Check multiple rate limiting windows."""
        for window_name, window_seconds in self.rate_windows.items():
            key = f"ratelimit:{source_ip}:{window_name}"
            count = await self.redis.incr(key)
            
            if count == 1:
                await self.redis.expire(key, window_seconds)
                
            if self._is_rate_exceeded(window_name, count):
                return False
                
        return True

    def _is_rate_exceeded(self, window: str, count: int) -> bool:
        """Check if rate limit is exceeded for window."""
        limits = {
            "1m": 30,
            "5m": 100,
            "1h": 1000,
            "24h": 10000
        }
        return count > limits.get(window, 0)

    async def _detect_automated_threats(self, source_ip: str) -> bool:
        """Detect automated threats and bot behavior."""
        key = f"requests:{source_ip}"
        
        # Check request patterns
        pattern_score = await self._calculate_pattern_score(key)
        if pattern_score > 0.8:  # Threshold for bot detection
            self._log_security_event(
                event_type="automated_threat",
                threat_level=ThreatLevel.HIGH,
                details={"pattern_score": pattern_score}
            )
            return True
            
        return False

    async def _detect_anomalies(self, payload: Dict) -> bool:
        """Detect anomalous request patterns."""
        # Implement anomaly detection logic
        return False

    def _validate_signature(self, request_id: str, payload: Dict) -> bool:
        """Validate request signature."""
        try:
            payload_bytes = str(payload).encode()
            expected_signature = base64.b64encode(
                hmac.new(
                    self.secret_key.encode(),
                    payload_bytes,
                    hashlib.sha256
                ).digest()
            ).decode()
            
            return hmac.compare_digest(
                expected_signature,
                payload.get("signature", "")
            )
        except Exception:
            return False

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        try:
            return self.fernet.encrypt(data.encode()).decode()
        except Exception as e:
            logger.error(f"Encryption error: {str(e)}")
            raise

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        try:
            return self.fernet.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            logger.error(f"Decryption error: {str(e)}")
            raise

    def generate_jwt_token(
        self,
        user_id: str,
        expiry_hours: int = 24
    ) -> str:
        """Generate JWT token."""
        try:
            payload = {
                "user_id": user_id,
                "exp": datetime.utcnow() + timedelta(hours=expiry_hours),
                "iat": datetime.utcnow()
            }
            return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        except Exception as e:
            logger.error(f"Token generation error: {str(e)}")
            raise

    def validate_jwt_token(self, token: str) -> Tuple[bool, Optional[Dict]]:
        """Validate JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=["HS256"]
            )
            return True, payload
        except jwt.ExpiredSignatureError:
            return False, {"error": "Token expired"}
        except jwt.InvalidTokenError:
            return False, {"error": "Invalid token"}

    def _log_security_event(
        self,
        event_type: str,
        threat_level: ThreatLevel,
        details: Dict,
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """Log security event."""
        event = SecurityEvent(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            threat_level=threat_level,
            details=details,
            source_ip=source_ip or "unknown",
            user_id=user_id
        )
        
        logger.warning(
            f"Security event: {event_type} "
            f"[{threat_level.value}] - {details}"
        )
        
        # Store event in Redis for analysis
        self.redis.lpush(
            "security_events",
            str(event.__dict__)
        )

    def _is_internal_request(self) -> bool:
        """Check if request is from internal network."""
        # Implement internal request detection
        return False

    async def analyze_security_events(
        self,
        time_window: int = 3600
    ) -> Dict:
        """Analyze security events for patterns."""
        events = await self.redis.lrange(
            "security_events",
            0,
            -1
        )
        
        # Implement security event analysis
        return {
            "total_events": len(events),
            "threat_levels": {},
            "event_types": {}
        }

    def update_security_rules(self, rules: Dict):
        """Update security rules dynamically."""
        # Update threat patterns
        if "threat_patterns" in rules:
            self.threat_patterns.update(rules["threat_patterns"])
            
        # Update IP lists
        if "ip_blacklist" in rules:
            self.ip_blacklist.update(rules["ip_blacklist"])
        if "ip_whitelist" in rules:
            self.ip_whitelist.update(rules["ip_whitelist"])
