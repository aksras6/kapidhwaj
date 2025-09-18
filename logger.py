"""
Centralized Logging System for Trading Application

Replaces all the print() statements in your original code with proper logging.
Provides different log levels, file output, and structured logging.
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import json


class TradingLogger:
    """Custom logger for trading applications with enhanced formatting"""
    
    def __init__(self, name: str, log_dir: str = "logs", level: int = logging.INFO):
        self.name = name
        self.log_dir = log_dir
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup file and console handlers with custom formatting"""
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Custom formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler - daily rotation
        today = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(self.log_dir, f"{self.name}_{today}.log")
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler with colors (if available)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(ColoredFormatter())
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Log startup
        self.logger.info(f"Logger '{self.name}' initialized. Log file: {log_file}")
    
    def debug(self, message: str, **kwargs):
        """Debug level logging"""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Info level logging"""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Warning level logging"""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Error level logging"""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Critical level logging"""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def log(self, level: int, message: str, **kwargs):
        """Log method for compatibility with LoggingContext"""
        self._log_with_context(level, message, **kwargs)
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log with additional context data"""
        if kwargs:
            context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            full_message = f"{message} | {context}"
        else:
            full_message = message
        
        self.logger.log(level, full_message)
    
    def trade_event(self, event_type: str, symbol: str, **data):
        """Special logging for trading events"""
        timestamp = datetime.now().isoformat()
        event_data = {
            'timestamp': timestamp,
            'event_type': event_type,
            'symbol': symbol,
            **data
        }
        
        # Log as info with structured format
        self.info(f"TRADE_EVENT: {event_type}", **event_data)
        
        # Also save to separate trade events file
        self._save_trade_event(event_data)
    
    def _save_trade_event(self, event_data: Dict[str, Any]):
        """Save trade events to separate JSON file for analysis"""
        events_file = os.path.join(self.log_dir, f"trade_events_{datetime.now().strftime('%Y%m%d')}.jsonl")
        
        try:
            with open(events_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event_data) + '\n')
        except Exception as e:
            self.error(f"Failed to save trade event: {e}")


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(self):
        super().__init__(
            '%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def format(self, record):
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{level_color}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


# ===== LOGGER FACTORY =====

class LoggerFactory:
    """Factory for creating and managing loggers"""
    
    _loggers: Dict[str, TradingLogger] = {}
    _default_level = logging.INFO
    _default_log_dir = "logs"
    
    @classmethod
    def get_logger(cls, name: str, level: Optional[int] = None, log_dir: Optional[str] = None) -> TradingLogger:
        """Get or create a logger with given name"""
        
        if name not in cls._loggers:
            actual_level = level or cls._default_level
            actual_log_dir = log_dir or cls._default_log_dir
            cls._loggers[name] = TradingLogger(name, actual_log_dir, actual_level)
        
        return cls._loggers[name]
    
    @classmethod
    def set_global_level(cls, level: int):
        """Set logging level for all existing loggers"""
        cls._default_level = level
        for logger in cls._loggers.values():
            logger.logger.setLevel(level)
    
    @classmethod
    def set_global_log_dir(cls, log_dir: str):
        """Set log directory for future loggers"""
        cls._default_log_dir = log_dir


# ===== CONVENIENCE FUNCTIONS =====

def get_logger(name: str) -> TradingLogger:
    """Convenience function to get a logger"""
    return LoggerFactory.get_logger(name)


def setup_logging(level: int = logging.INFO, log_dir: str = "logs"):
    """Setup global logging configuration"""
    LoggerFactory.set_global_level(level)
    LoggerFactory.set_global_log_dir(log_dir)
    
    # Create main application logger
    main_logger = get_logger("trading_system")
    main_logger.info("Logging system initialized")
    
    return main_logger


# ===== TRADING-SPECIFIC LOGGERS =====

def get_data_logger() -> TradingLogger:
    """Get logger for data operations"""
    return get_logger("data")


def get_signals_logger() -> TradingLogger:
    """Get logger for signal generation"""
    return get_logger("signals")


def get_orders_logger() -> TradingLogger:
    """Get logger for order management"""
    return get_logger("orders")


def get_backtest_logger() -> TradingLogger:
    """Get logger for backtesting"""
    return get_logger("backtest")


def get_ib_logger() -> TradingLogger:
    """Get logger for IB API operations"""
    return get_logger("ib_api")


# ===== PERFORMANCE MONITORING =====

class PerformanceLogger:
    """Logger for performance monitoring and timing"""
    
    def __init__(self, logger: TradingLogger):
        self.logger = logger
        self._timers = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self._timers[operation] = datetime.now()
        self.logger.debug(f"Started: {operation}")
    
    def end_timer(self, operation: str):
        """End timing and log duration"""
        if operation in self._timers:
            duration = (datetime.now() - self._timers[operation]).total_seconds()
            self.logger.info(f"Completed: {operation}", duration_sec=duration)
            del self._timers[operation]
        else:
            self.logger.warning(f"Timer not found for operation: {operation}")
    
    def log_memory_usage(self):
        """Log current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.info("Memory usage", memory_mb=round(memory_mb, 2))
        except ImportError:
            self.logger.debug("psutil not available for memory monitoring")
    
    def log_data_stats(self, data, name: str):
        """Log statistics about data (DataFrame, list, etc.)"""
        if hasattr(data, 'shape'):  # DataFrame or numpy array
            self.logger.info(f"Data stats: {name}", shape=data.shape, memory_mb=round(data.memory_usage(deep=True).sum() / 1024**2, 2))
        elif hasattr(data, '__len__'):  # List or similar
            self.logger.info(f"Data stats: {name}", length=len(data))
        else:
            self.logger.debug(f"Data stats: {name}", type=type(data).__name__)


# ===== CONTEXT MANAGERS =====

class LoggingContext:
    """Context manager for operation logging"""
    
    def __init__(self, logger: TradingLogger, operation: str, level: int = logging.INFO):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log(self.level, f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.log(self.level, f"Completed: {self.operation}", duration_sec=duration)
        else:
            self.logger.error(f"Failed: {self.operation}", duration_sec=duration, error=str(exc_val))


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    # Test the logging system
    print("=== Testing Logging System ===")
    
    # Setup logging
    main_logger = setup_logging(level=logging.DEBUG)
    
    # Test different loggers
    data_logger = get_data_logger()
    signals_logger = get_signals_logger()
    orders_logger = get_orders_logger()
    
    # Test basic logging
    main_logger.info("System startup")
    data_logger.debug("Fetching historical data", symbol="AAPL", bars=100)
    signals_logger.warning("ADX threshold not met", symbol="MSFT", adx=25.0)
    orders_logger.error("Order placement failed", symbol="GOOGL", reason="insufficient_margin")
    
    # Test trade event logging
    main_logger.trade_event("ENTRY", "AAPL", 
                           shares=100, 
                           price=150.25, 
                           stop_loss=145.00,
                           signal_strength=0.85)
    
    # Test performance logging
    perf_logger = PerformanceLogger(main_logger)
    perf_logger.start_timer("data_processing")
    
    # Simulate some work
    import time
    time.sleep(0.1)
    
    perf_logger.end_timer("data_processing")
    perf_logger.log_memory_usage()
    
    # Test context manager
    with LoggingContext(main_logger, "complex_operation"):
        time.sleep(0.05)
        main_logger.info("Inside complex operation")
    
    main_logger.info("Logging test complete")
    print("\n=== Check the logs directory for output files ===")