"""API middleware for request processing and error handling"""

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from typing import Callable
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class RequestMiddleware:
    """Middleware for request processing"""
    
    async def __call__(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request"""
        start_time = time.time()
        
        # Add request ID
        request.state.request_id = f"{int(time.time()*1000)}"
        
        try:
            # Process request
            response = await call_next(request)
            
            # Add processing time header
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            
            # Log request
            self._log_request(request, response, process_time)
            
            return response
            
        except Exception as e:
            # Log error
            logger.error(
                f"Request failed: {str(e)}",
                extra={
                    "request_id": request.state.request_id,
                    "path": request.url.path,
                    "method": request.method
                }
            )
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "detail": str(e),
                    "request_id": request.state.request_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    def _log_request(
        self,
        request: Request,
        response: Response,
        process_time: float
    ):
        """Log request details"""
        logger.info(
            f"{request.method} {request.url.path}",
            extra={
                "request_id": request.state.request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "process_time": process_time,
                "user_agent": request.headers.get("user-agent"),
                "ip": request.client.host
            }
        )

class CacheMiddleware:
    """Middleware for response caching"""
    
    def __init__(self):
        self.cache = {}
        self.max_age = 300  # 5 minutes
        
    async def __call__(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request with caching"""
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        cache_key = f"{request.method}:{request.url.path}:{request.query_params}"
        
        # Check cache
        cached = self.cache.get(cache_key)
        if cached:
            age = time.time() - cached["timestamp"]
            if age < self.max_age:
                response = cached["response"]
                response.headers["X-Cache"] = "HIT"
                response.headers["Age"] = str(int(age))
                return response
        
        # Process request
        response = await call_next(request)
        
        # Cache response
        if 200 <= response.status_code < 300:
            self.cache[cache_key] = {
                "response": response,
                "timestamp": time.time()
            }
            response.headers["X-Cache"] = "MISS"
        
        return response

class CompressionMiddleware:
    """Middleware for response compression"""
    
    async def __call__(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request with compression"""
        response = await call_next(request)
        
        # Check if client accepts compression
        accepts_encoding = request.headers.get("Accept-Encoding", "")
        
        if "gzip" in accepts_encoding:
            # Add compression headers
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Vary"] = "Accept-Encoding"
        
        return response
