"""
RA.Aid WebUI - Logging Module

This module handles logging configuration for the WebUI component.
"""

import logging
import sys
from datetime import datetime

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and emojis for different log levels"""
    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',   # Green
        'WARNING': '\033[33m', # Yellow
        'ERROR': '\033[31m',   # Red
        'CRITICAL': '\033[41m' # Red background
    }
    EMOJIS = {
        'DEBUG': '🔍',
        'INFO': '📝',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }
    RESET = '\033[0m'

    def format(self, record):
        # Add emoji and color based on log level
        color = self.COLORS.get(record.levelname, '')
        emoji = self.EMOJIS.get(record.levelname, '')
        reset = self.RESET
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        # Format the message
        message = super().format(record)
        return f"{color}{emoji} [{timestamp}] {record.levelname:<8} | {message}{reset}"

def setup_logger(name='webui', level=logging.INFO):
    """
    Configure logging with a clean, informative format
    
    Args:
        name (str): Logger name
        level (int): Logging level
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create console handler with custom formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter('%(message)s'))
    logger.addHandler(console_handler)
    
    return logger 