import logging
import sys
import os
from datetime import datetime
from typing import Optional
import json

class CustomFormatter(logging.Formatter):
    """Custom formatter for colored and structured logs"""
    
    # ANSI color codes
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    # Log format with colors
    format_colored = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    format_plain = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Level-specific formats
    FORMATS = {
        logging.DEBUG: grey + format_colored + reset,
        logging.INFO: green + format_colored + reset,
        logging.WARNING: yellow + format_colored + reset,
        logging.ERROR: red + format_colored + reset,
        logging.CRITICAL: bold_red + format_colored + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.format_plain)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": record.process,
            "thread_id": record.thread
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)
        
        return json.dumps(log_entry)

def setup_logger(
    name: str = "model_training_service",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    json_format: bool = False,
    enable_console: bool = True
) -> logging.Logger:
    """
    Set up and configure logger
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to write logs to
        json_format: Whether to use JSON formatting
        enable_console: Whether to output to console
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = CustomFormatter()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def get_logger(name: str = None) -> logging.Logger:
    """
    Get or create logger with default configuration
    
    Args:
        name: Logger name (uses caller's module name if None)
    
    Returns:
        Logger instance
    """
    if name is None:
        # Get caller's module name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return logging.getLogger(name)

class TrainingLogger:
    """Specialized logger for model training operations"""
    
    def __init__(self, logger_name: str = "training"):
        self.logger = get_logger(logger_name)
        self.training_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def log_training_start(self, model_name: str, dataset_info: dict):
        """Log training session start"""
        self.logger.info(
            f"🚀 Starting training for {model_name}",
            extra={'extra_data': {
                'training_id': self.training_id,
                'model_name': model_name,
                'dataset_info': dataset_info,
                'event_type': 'training_start'
            }}
        )
    
    def log_training_progress(self, model_name: str, epoch: int, total_epochs: int, metrics: dict):
        """Log training progress"""
        progress = (epoch / total_epochs) * 100
        self.logger.info(
            f"📊 {model_name} - Epoch {epoch}/{total_epochs} ({progress:.1f}%)",
            extra={'extra_data': {
                'training_id': self.training_id,
                'model_name': model_name,
                'epoch': epoch,
                'total_epochs': total_epochs,
                'progress_percent': progress,
                'metrics': metrics,
                'event_type': 'training_progress'
            }}
        )
    
    def log_training_complete(self, model_name: str, final_metrics: dict, duration: float):
        """Log training completion"""
        self.logger.info(
            f"✅ Training completed for {model_name} in {duration:.2f}s",
            extra={'extra_data': {
                'training_id': self.training_id,
                'model_name': model_name,
                'final_metrics': final_metrics,
                'duration_seconds': duration,
                'event_type': 'training_complete'
            }}
        )
    
    def log_model_saved(self, model_name: str, file_path: str, file_size: int):
        """Log model save operation"""
        self.logger.info(
            f"💾 Model saved: {model_name} ({file_size} bytes)",
            extra={'extra_data': {
                'training_id': self.training_id,
                'model_name': model_name,
                'file_path': file_path,
                'file_size_bytes': file_size,
                'event_type': 'model_saved'
            }}
        )
    
    def log_error(self, operation: str, error: Exception, context: dict = None):
        """Log error with context"""
        self.logger.error(
            f"❌ Error in {operation}: {str(error)}",
            extra={'extra_data': {
                'training_id': self.training_id,
                'operation': operation,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context or {},
                'event_type': 'error'
            }},
            exc_info=True
        )
    
    def log_data_loading(self, dataset_name: str, record_count: int, features_count: int):
        """Log dataset loading information"""
        self.logger.info(
            f"📥 Loaded {dataset_name}: {record_count} records, {features_count} features",
            extra={'extra_data': {
                'training_id': self.training_id,
                'dataset_name': dataset_name,
                'record_count': record_count,
                'features_count': features_count,
                'event_type': 'data_loaded'
            }}
        )

# Convenience functions
def log_model_metrics(model_name: str, metrics: dict):
    """Convenience function to log model metrics"""
    logger = get_logger("model_metrics")
    logger.info(
        f"📈 {model_name} metrics",
        extra={'extra_data': {
            'model_name': model_name,
            'metrics': metrics,
            'event_type': 'model_metrics'
        }}
    )

def log_prediction_batch(model_name: str, batch_size: int, processing_time: float):
    """Convenience function to log prediction batch"""
    logger = get_logger("predictions")
    logger.info(
        f"🔮 {model_name} processed {batch_size} predictions in {processing_time:.3f}s",
        extra={'extra_data': {
            'model_name': model_name,
            'batch_size': batch_size,
            'processing_time_seconds': processing_time,
            'event_type': 'prediction_batch'
        }}
    )

def log_feature_importance(model_name: str, top_features: list):
    """Convenience function to log feature importance"""
    logger = get_logger("feature_analysis")
    logger.info(
        f"🎯 {model_name} top features",
        extra={'extra_data': {
            'model_name': model_name,
            'top_features': top_features,
            'event_type': 'feature_importance'
        }}
    )

# Initialize default logger when module is imported
default_logger = setup_logger()

# Export commonly used functions
__all__ = [
    'setup_logger',
    'get_logger',
    'TrainingLogger',
    'log_model_metrics',
    'log_prediction_batch',
    'log_feature_importance',
    'default_logger'
]