"""Request batching implementation for API endpoints."""
import asyncio
from typing import Any, List, Optional, Dict
import logging
from datetime import datetime
import numpy as np

from ..core.utils.performance_monitoring import BatchProcessor

logger = logging.getLogger(__name__)

class RequestBatcher:
    """Batch incoming requests for efficient processing."""
    
    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_time: float = 0.1,
        max_retries: int = 3
    ):
        self.batch_processor = BatchProcessor(max_batch_size, max_wait_time)
        self.max_retries = max_retries
        self.processing = False
        self._lock = asyncio.Lock()
        
    async def process_request(
        self,
        request_data: Dict,
        processor_func: Any
    ) -> Any:
        """Add request to batch and process when ready."""
        async with self._lock:
            should_process = await self.batch_processor.add_to_batch(request_data)
            
            if should_process and not self.processing:
                self.processing = True
                try:
                    results = await self._process_batch(processor_func)
                    return results
                finally:
                    self.processing = False
            return None
            
    async def _process_batch(self, processor_func: Any) -> List:
        """Process the current batch with retries."""
        retries = 0
        while retries < self.max_retries:
            try:
                return await self.batch_processor.process_batch(processor_func)
            except Exception as e:
                retries += 1
                if retries == self.max_retries:
                    logger.error(
                        f"Failed to process batch after {retries} attempts: {str(e)}"
                    )
                    raise
                await asyncio.sleep(0.1 * (2 ** retries))  # Exponential backoff

class ModelBatcher:
    """Batch model inference requests."""
    
    def __init__(
        self,
        model_name: str,
        optimal_batch_size: int = 32,
        max_wait_time: float = 0.1
    ):
        self.model_name = model_name
        self.request_batcher = RequestBatcher(optimal_batch_size, max_wait_time)
        self.batch_sizes: List[int] = []
        self.processing_times: List[float] = []
        
    async def predict(
        self,
        input_data: Dict,
        model_func: Any
    ) -> Any:
        """Add prediction request to batch."""
        start_time = datetime.now()
        
        result = await self.request_batcher.process_request(
            input_data,
            model_func
        )
        
        if result is not None:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.processing_times.append(processing_time)
            self.batch_sizes.append(len(result))
            
        return result
        
    def get_batch_stats(self) -> Dict:
        """Get batching performance statistics."""
        if not self.batch_sizes:
            return {}
            
        return {
            "model_name": self.model_name,
            "avg_batch_size": np.mean(self.batch_sizes),
            "avg_processing_time": np.mean(self.processing_times),
            "p95_processing_time": np.percentile(self.processing_times, 95),
            "total_batches": len(self.batch_sizes)
        }

class EndpointBatcher:
    """Batch endpoint requests for improved throughput."""
    
    def __init__(
        self,
        endpoint_name: str,
        max_batch_size: int = 50,
        max_wait_time: float = 0.2
    ):
        self.endpoint_name = endpoint_name
        self.request_batcher = RequestBatcher(max_batch_size, max_wait_time)
        self.request_counts: Dict[str, int] = {}
        self.batch_stats: List[Dict] = []
        
    async def process_endpoint_request(
        self,
        request_data: Dict,
        endpoint_func: Any
    ) -> Any:
        """Process endpoint request in batch."""
        # Track request type
        request_type = request_data.get("type", "default")
        self.request_counts[request_type] = (
            self.request_counts.get(request_type, 0) + 1
        )
        
        start_time = datetime.now()
        
        result = await self.request_batcher.process_request(
            request_data,
            endpoint_func
        )
        
        if result is not None:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.batch_stats.append({
                "timestamp": start_time,
                "batch_size": len(result),
                "processing_time": processing_time
            })
            
        return result
        
    def get_endpoint_stats(self) -> Dict:
        """Get endpoint batching statistics."""
        if not self.batch_stats:
            return {
                "endpoint_name": self.endpoint_name,
                "request_counts": self.request_counts,
                "total_requests": sum(self.request_counts.values())
            }
            
        processing_times = [stat["processing_time"] for stat in self.batch_stats]
        batch_sizes = [stat["batch_size"] for stat in self.batch_stats]
        
        return {
            "endpoint_name": self.endpoint_name,
            "request_counts": self.request_counts,
            "total_requests": sum(self.request_counts.values()),
            "avg_batch_size": np.mean(batch_sizes),
            "avg_processing_time": np.mean(processing_times),
            "p95_processing_time": np.percentile(processing_times, 95),
            "total_batches": len(self.batch_stats)
        }
