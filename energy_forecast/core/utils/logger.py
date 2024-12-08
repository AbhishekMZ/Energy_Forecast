import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional

class StructuredLogger:
    def __init__(self, name: str, log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Console handler with structured format
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(console_handler)
        
        # File handler for persistent logs
        file_handler = logging.FileHandler("logs/energy_forecast.log")
        file_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(file_handler)

    def _log(self, level: str, message: str, **kwargs):
        extra = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": "energy_forecast",
            **kwargs
        }
        getattr(self.logger, level.lower())(message, extra={"structured": extra})

    def info(self, message: str, **kwargs):
        self._log("INFO", message, **kwargs)

    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        extra = kwargs
        if error:
            extra.update({
                "error_type": error.__class__.__name__,
                "error_message": str(error)
            })
        self._log("ERROR", message, **extra)

    def warning(self, message: str, **kwargs):
        self._log("WARNING", message, **kwargs)

    def debug(self, message: str, **kwargs):
        self._log("DEBUG", message, **kwargs)

class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # Get basic log information
        log_data = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name
        }
        
        # Add structured data if available
        if hasattr(record, "structured"):
            log_data.update(record.structured)
        
        return json.dumps(log_data)
