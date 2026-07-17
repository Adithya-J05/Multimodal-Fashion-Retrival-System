"""Structured logging and profiling setup."""

import logging
import sys
from typing import Optional
from datetime import datetime
import json


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage()
        }
        
        if hasattr(record, "extra_data"):
            log_entry["extra"] = record.extra_data
            
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False
) -> logging.Logger:
    """
    Setup logger with console and optional file handlers.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        json_format: Use JSON format for logs
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
        
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
        logger.addHandler(file_handler)
    
    return logger


class LoggerContext:
    """Context manager for logging with extra data."""
    
    def __init__(self, logger: logging.Logger, extra_data: dict):
        self.logger = logger
        self.extra_data = extra_data
        
    def __enter__(self):
        self.logger.debug("Entering context", extra={"extra_data": self.extra_data})
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.logger.error(
                f"Context exited with error: {exc_val}",
                extra={"extra_data": self.extra_data}
            )
        else:
            self.logger.debug("Exiting context", extra={"extra_data": self.extra_data})


def get_logger(name: str, **kwargs) -> logging.Logger:
    """Convenience function to get a logger instance."""
    return setup_logger(name, **kwargs)