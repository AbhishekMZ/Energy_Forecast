"""Caching implementation for API responses and model predictions."""
from typing import Any, Optional, Union, Dict
import asyncio
from datetime import datetime, timedelta
import json
import hashlib
import logging
from redis import asyncio as aioredis
from functools import wraps

from ..core.utils.performance_monitoring import CacheProfiler

logger = logging.getLogger(__name__)

class CacheManager:
    """Manage caching for API responses and model predictions."""
    
    def __init__(
        self,
        redis_url: str,
        default_ttl: int = 3600,
        max_batch_size: int = 1000
    ):
        self.redis = aioredis.from_url(redis_url)
        self.default_ttl = default_ttl
        self.max_batch_size = max_batch_size
        self.profiler = CacheProfiler("redis_cache")
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = await self.redis.get(key)
            hit = value is not None
            self.profiler.record_cache_access(hit)
            
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None
            
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        try:
            ttl = ttl or self.default_ttl
            serialized = json.dumps(value)
            await self.redis.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
            return False
            
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {str(e)}")
            return False
            
    async def get_many(self, keys: list) -> Dict[str, Any]:
        """Get multiple values from cache."""
        try:
            pipe = self.redis.pipeline()
            for key in keys:
                pipe.get(key)
            
            values = await pipe.execute()
            result = {}
            
            for key, value in zip(keys, values):
                if value:
                    result[key] = json.loads(value)
                    self.profiler.record_cache_access(True)
                else:
                    self.profiler.record_cache_access(False)
                    
            return result
        except Exception as e:
            logger.error(f"Cache get_many error: {str(e)}")
            return {}
            
    async def set_many(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Set multiple values in cache."""
        try:
            ttl = ttl or self.default_ttl
            pipe = self.redis.pipeline()
            
            for key, value in mapping.items():
                serialized = json.dumps(value)
                pipe.setex(key, ttl, serialized)
                
            await pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Cache set_many error: {str(e)}")
            return False
            
    def cache_response(
        self,
        prefix: str,
        ttl: Optional[int] = None
    ):
        """Decorator to cache API responses."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                key_parts = [prefix]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5(
                    ":".join(key_parts).encode()
                ).hexdigest()
                
                # Try to get from cache
                cached = await self.get(cache_key)
                if cached is not None:
                    return cached
                    
                # Generate and cache response
                response = await func(*args, **kwargs)
                await self.set(cache_key, response, ttl)
                return response
                
            return wrapper
        return decorator
        
    def cache_model_prediction(
        self,
        model_name: str,
        ttl: Optional[int] = None
    ):
        """Decorator to cache model predictions."""
        def decorator(func):
            @wraps(func)
            async def wrapper(inputs, *args, **kwargs):
                if not isinstance(inputs, list):
                    inputs = [inputs]
                    
                # Generate cache keys
                cache_keys = []
                uncached_indices = []
                uncached_inputs = []
                
                for i, input_data in enumerate(inputs):
                    key_parts = [model_name]
                    key_parts.extend(
                        f"{k}:{v}" for k, v in sorted(input_data.items())
                    )
                    cache_key = hashlib.md5(
                        ":".join(key_parts).encode()
                    ).hexdigest()
                    cache_keys.append(cache_key)
                    
                # Get cached predictions
                cached_results = await self.get_many(cache_keys)
                predictions = [None] * len(inputs)
                
                for i, cache_key in enumerate(cache_keys):
                    if cache_key in cached_results:
                        predictions[i] = cached_results[cache_key]
                    else:
                        uncached_indices.append(i)
                        uncached_inputs.append(inputs[i])
                        
                # Generate predictions for uncached inputs
                if uncached_inputs:
                    new_predictions = await func(uncached_inputs, *args, **kwargs)
                    
                    # Cache new predictions
                    cache_updates = {}
                    for i, pred in zip(uncached_indices, new_predictions):
                        predictions[i] = pred
                        cache_updates[cache_keys[i]] = pred
                        
                    await self.set_many(cache_updates, ttl)
                    
                return predictions[0] if len(inputs) == 1 else predictions
                
            return wrapper
        return decorator
