"""Security middleware for the Energy Forecast Platform."""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import jwt
import time
import re
import hashlib
from typing import Optional, Dict, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SecurityMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        secret_key: str,
        rate_limit: int = 100,
        rate_window: int = 3600,
        allowed_hosts: List[str] = None,
        cors_origins: List[str] = None
    ):
        super().__init__(app)
        self.secret_key = secret_key
        self.rate_limit = rate_limit
        self.rate_window = rate_window
        self.allowed_hosts = allowed_hosts or ["localhost", "127.0.0.1"]
        self.cors_origins = cors_origins or []
        self.request_counts: Dict[str, List[float]] = {}
        
        # Regex patterns for input validation
        self.sql_injection_pattern = re.compile(
            r"(\b(select|insert|update|delete|drop|union|exec)\b)", 
            re.IGNORECASE
        )
        self.xss_pattern = re.compile(
            r"<[^>]*script|javascript:|data:|vbscript:|onload=|onerror=",
            re.IGNORECASE
        )

    async def dispatch(self, request: Request, call_next) -> Response:
        """Main middleware dispatch method."""
        try:
            # Pre-request security checks
            if not await self._pre_request_checks(request):
                return JSONResponse(
                    status_code=403,
                    content={"error": "Security check failed"}
                )

            # Process request
            response = await call_next(request)

            # Post-request security headers
            return self._add_security_headers(response)

        except Exception as e:
            logger.error(f"Security middleware error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal security error"}
            )

    async def _pre_request_checks(self, request: Request) -> bool:
        """Perform pre-request security checks."""
        return all([
            self._check_host(request),
            await self._check_rate_limit(request),
            self._validate_token(request),
            await self._validate_input(request),
            self._check_cors(request)
        ])

    def _check_host(self, request: Request) -> bool:
        """Validate host header."""
        host = request.headers.get("host", "").split(":")[0]
        if host not in self.allowed_hosts:
            logger.warning(f"Invalid host detected: {host}")
            return False
        return True

    async def _check_rate_limit(self, request: Request) -> bool:
        """Check rate limiting."""
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old requests
        if client_ip in self.request_counts:
            self.request_counts[client_ip] = [
                t for t in self.request_counts[client_ip]
                if current_time - t < self.rate_window
            ]
        else:
            self.request_counts[client_ip] = []

        # Check rate limit
        if len(self.request_counts[client_ip]) >= self.rate_limit:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return False

        self.request_counts[client_ip].append(current_time)
        return True

    def _validate_token(self, request: Request) -> bool:
        """Validate JWT token."""
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return False

        token = auth_header.split(" ")[1]
        try:
            jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return True
        except jwt.InvalidTokenError:
            logger.warning("Invalid token detected")
            return False

    async def _validate_input(self, request: Request) -> bool:
        """Validate and sanitize input."""
        # Check query parameters
        for key, value in request.query_params.items():
            if not self._is_safe_input(value):
                logger.warning(f"Suspicious query parameter detected: {key}={value}")
                return False

        # Check request body if it's JSON
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.json()
                if not self._validate_json_input(body):
                    return False
            except ValueError:
                pass

        return True

    def _is_safe_input(self, value: str) -> bool:
        """Check if input is safe from SQL injection and XSS."""
        if self.sql_injection_pattern.search(value):
            return False
        if self.xss_pattern.search(value):
            return False
        return True

    def _validate_json_input(self, data: dict, depth: int = 0) -> bool:
        """Recursively validate JSON input."""
        if depth > 10:  # Prevent deep recursion
            return False

        if isinstance(data, dict):
            return all(
                isinstance(k, str) and 
                self._is_safe_input(k) and 
                self._validate_json_input(v, depth + 1)
                for k, v in data.items()
            )
        elif isinstance(data, list):
            return all(
                self._validate_json_input(item, depth + 1)
                for item in data
            )
        elif isinstance(data, str):
            return self._is_safe_input(data)
        return True

    def _check_cors(self, request: Request) -> bool:
        """Check CORS headers."""
        origin = request.headers.get("origin")
        if origin and origin not in self.cors_origins:
            logger.warning(f"Invalid CORS origin: {origin}")
            return False
        return True

    def _add_security_headers(self, response: Response) -> Response:
        """Add security headers to response."""
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": self._generate_csp(),
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            "Cache-Control": "no-store, max-age=0",
        }
        
        for key, value in headers.items():
            response.headers[key] = value

        return response

    def _generate_csp(self) -> str:
        """Generate Content Security Policy."""
        return (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self'; "
            "img-src 'self' data:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "media-src 'none'; "
            "object-src 'none'; "
            "frame-src 'none'; "
            "worker-src 'self'; "
            "form-action 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "manifest-src 'self'"
        )

    def _get_request_id(self, request: Request) -> str:
        """Generate unique request ID."""
        timestamp = datetime.now().isoformat()
        client_ip = request.client.host
        path = request.url.path
        return hashlib.sha256(
            f"{timestamp}{client_ip}{path}".encode()
        ).hexdigest()[:16]
