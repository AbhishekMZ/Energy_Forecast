"""Advanced threat detection system for the Energy Forecast Platform."""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import json
import redis
from sklearn.ensemble import IsolationForest
from collections import defaultdict
import ipaddress
import re

logger = logging.getLogger(__name__)

class ThreatType(Enum):
    BRUTE_FORCE = "brute_force"
    DDOS = "ddos"
    DATA_EXFILTRATION = "data_exfiltration"
    INJECTION = "injection"
    ANOMALY = "anomaly"
    BOT = "bot"

@dataclass
class ThreatEvent:
    timestamp: datetime
    threat_type: ThreatType
    source_ip: str
    confidence: float
    details: Dict
    request_id: str

class ThreatDetectionSystem:
    def __init__(
        self,
        redis_client: redis.Redis,
        config: Dict
    ):
        self.redis = redis_client
        self.config = config
        self._init_detection_systems()

    def _init_detection_systems(self):
        """Initialize detection subsystems."""
        # Anomaly detection model
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )

        # Rate limiting windows
        self.rate_windows = {
            "1m": 60,
            "5m": 300,
            "1h": 3600
        }

        # Pattern matching
        self.patterns = {
            "sql_injection": re.compile(
                r"(\b(union|select|insert|delete|drop|exec)\b)",
                re.IGNORECASE
            ),
            "xss": re.compile(
                r"(<script|javascript:|data:text/html|<img|onerror=)",
                re.IGNORECASE
            ),
            "path_traversal": re.compile(r"\.\.\/|\.\.\\"),
            "command_injection": re.compile(r"[;&|`]"),
            "email_pattern": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
        }

    async def analyze_request(
        self,
        request_data: Dict,
        request_id: str
    ) -> List[ThreatEvent]:
        """Analyze request for potential threats."""
        threats = []
        source_ip = request_data.get("source_ip")

        # Run all threat detection methods
        detection_methods = [
            self._detect_brute_force,
            self._detect_ddos,
            self._detect_data_exfiltration,
            self._detect_injection,
            self._detect_anomalies,
            self._detect_bot_behavior
        ]

        for method in detection_methods:
            threat = await method(request_data, request_id)
            if threat:
                threats.append(threat)
                await self._handle_threat(threat)

        return threats

    async def _detect_brute_force(
        self,
        request_data: Dict,
        request_id: str
    ) -> Optional[ThreatEvent]:
        """Detect brute force attempts."""
        source_ip = request_data.get("source_ip")
        endpoint = request_data.get("endpoint")
        
        if endpoint != "/api/v1/auth":
            return None

        # Check failed attempts
        key = f"failed_auth:{source_ip}"
        failed_attempts = await self.redis.incr(key)
        await self.redis.expire(key, 3600)  # 1 hour window

        if failed_attempts > self.config["max_auth_attempts"]:
            return ThreatEvent(
                timestamp=datetime.utcnow(),
                threat_type=ThreatType.BRUTE_FORCE,
                source_ip=source_ip,
                confidence=0.9,
                details={
                    "failed_attempts": failed_attempts,
                    "window": "1h"
                },
                request_id=request_id
            )

        return None

    async def _detect_ddos(
        self,
        request_data: Dict,
        request_id: str
    ) -> Optional[ThreatEvent]:
        """Detect DDoS attacks."""
        source_ip = request_data.get("source_ip")
        
        # Check request rates across different windows
        for window_name, window_seconds in self.rate_windows.items():
            key = f"requests:{source_ip}:{window_name}"
            request_count = await self.redis.incr(key)
            await self.redis.expire(key, window_seconds)

            threshold = self.config[f"ddos_threshold_{window_name}"]
            if request_count > threshold:
                return ThreatEvent(
                    timestamp=datetime.utcnow(),
                    threat_type=ThreatType.DDOS,
                    source_ip=source_ip,
                    confidence=min(request_count / threshold, 1.0),
                    details={
                        "request_count": request_count,
                        "window": window_name,
                        "threshold": threshold
                    },
                    request_id=request_id
                )

        return None

    async def _detect_data_exfiltration(
        self,
        request_data: Dict,
        request_id: str
    ) -> Optional[ThreatEvent]:
        """Detect potential data exfiltration."""
        source_ip = request_data.get("source_ip")
        payload = request_data.get("payload", {})
        
        # Check data volume
        data_size = len(json.dumps(payload))
        key = f"data_transfer:{source_ip}:1h"
        total_size = await self.redis.incrby(key, data_size)
        await self.redis.expire(key, 3600)

        if total_size > self.config["max_data_transfer"]:
            return ThreatEvent(
                timestamp=datetime.utcnow(),
                threat_type=ThreatType.DATA_EXFILTRATION,
                source_ip=source_ip,
                confidence=0.8,
                details={
                    "data_size": data_size,
                    "total_size": total_size,
                    "window": "1h"
                },
                request_id=request_id
            )

        # Check for sensitive data patterns
        if self._contains_sensitive_data(payload):
            return ThreatEvent(
                timestamp=datetime.utcnow(),
                threat_type=ThreatType.DATA_EXFILTRATION,
                source_ip=source_ip,
                confidence=0.9,
                details={
                    "reason": "sensitive_data_detected"
                },
                request_id=request_id
            )

        return None

    async def _detect_injection(
        self,
        request_data: Dict,
        request_id: str
    ) -> Optional[ThreatEvent]:
        """Detect injection attempts."""
        source_ip = request_data.get("source_ip")
        payload_str = json.dumps(request_data.get("payload", {}))
        
        for pattern_name, pattern in self.patterns.items():
            if pattern.search(payload_str):
                return ThreatEvent(
                    timestamp=datetime.utcnow(),
                    threat_type=ThreatType.INJECTION,
                    source_ip=source_ip,
                    confidence=0.95,
                    details={
                        "pattern_type": pattern_name,
                        "matched_pattern": pattern.pattern
                    },
                    request_id=request_id
                )

        return None

    async def _detect_anomalies(
        self,
        request_data: Dict,
        request_id: str
    ) -> Optional[ThreatEvent]:
        """Detect anomalous behavior."""
        # Extract features
        features = self._extract_features(request_data)
        
        # Update model periodically
        await self._update_anomaly_model()
        
        # Predict anomaly
        score = self.anomaly_detector.score_samples([features])[0]
        if score < self.config["anomaly_threshold"]:
            return ThreatEvent(
                timestamp=datetime.utcnow(),
                threat_type=ThreatType.ANOMALY,
                source_ip=request_data.get("source_ip"),
                confidence=1 - (score / self.config["anomaly_threshold"]),
                details={
                    "anomaly_score": score,
                    "threshold": self.config["anomaly_threshold"]
                },
                request_id=request_id
            )

        return None

    async def _detect_bot_behavior(
        self,
        request_data: Dict,
        request_id: str
    ) -> Optional[ThreatEvent]:
        """Detect automated bot behavior."""
        source_ip = request_data.get("source_ip")
        user_agent = request_data.get("headers", {}).get("user-agent", "")
        
        # Check known bot patterns
        if self._is_known_bot(user_agent):
            return None  # Allow known good bots

        # Check behavior patterns
        bot_score = await self._calculate_bot_score(source_ip, request_data)
        if bot_score > self.config["bot_threshold"]:
            return ThreatEvent(
                timestamp=datetime.utcnow(),
                threat_type=ThreatType.BOT,
                source_ip=source_ip,
                confidence=bot_score,
                details={
                    "bot_score": bot_score,
                    "user_agent": user_agent
                },
                request_id=request_id
            )

        return None

    async def _handle_threat(self, threat: ThreatEvent):
        """Handle detected threats."""
        # Log threat
        logger.warning(
            f"Threat detected: {threat.threat_type.value} "
            f"from {threat.source_ip} "
            f"(confidence: {threat.confidence})"
        )

        # Store in Redis for analysis
        await self.redis.lpush(
            "threat_events",
            json.dumps(threat.__dict__)
        )

        # Apply immediate actions based on threat type
        actions = {
            ThreatType.BRUTE_FORCE: self._handle_brute_force,
            ThreatType.DDOS: self._handle_ddos,
            ThreatType.DATA_EXFILTRATION: self._handle_data_exfiltration,
            ThreatType.INJECTION: self._handle_injection,
            ThreatType.ANOMALY: self._handle_anomaly,
            ThreatType.BOT: self._handle_bot
        }

        if threat.threat_type in actions:
            await actions[threat.threat_type](threat)

    def _contains_sensitive_data(self, payload: Dict) -> bool:
        """Check for sensitive data patterns."""
        payload_str = json.dumps(payload)
        
        # Check for common sensitive data patterns
        patterns = {
            "email": self.patterns["email_pattern"],
            "credit_card": re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "api_key": re.compile(r"[a-zA-Z0-9]{32,}")
        }

        return any(
            pattern.search(payload_str)
            for pattern in patterns.values()
        )

    def _extract_features(self, request_data: Dict) -> np.ndarray:
        """Extract features for anomaly detection."""
        # Implementation here
        return np.array([0])  # Placeholder

    async def _update_anomaly_model(self):
        """Periodically update anomaly detection model."""
        # Implementation here
        pass

    def _is_known_bot(self, user_agent: str) -> bool:
        """Check if user agent is a known good bot."""
        known_bots = {
            "googlebot",
            "bingbot",
            "yandexbot"
        }
        return any(bot in user_agent.lower() for bot in known_bots)

    async def _calculate_bot_score(
        self,
        source_ip: str,
        request_data: Dict
    ) -> float:
        """Calculate probability of bot behavior."""
        # Implementation here
        return 0.0  # Placeholder

    async def analyze_threats(
        self,
        time_window: int = 3600
    ) -> Dict:
        """Analyze recent threats."""
        threats = await self.redis.lrange(
            "threat_events",
            0,
            -1
        )
        
        analysis = defaultdict(int)
        for threat_data in threats:
            threat = json.loads(threat_data)
            analysis[threat["threat_type"]] += 1

        return dict(analysis)

    async def get_threat_stats(
        self,
        source_ip: Optional[str] = None
    ) -> Dict:
        """Get threat statistics."""
        # Implementation here
        return {}
