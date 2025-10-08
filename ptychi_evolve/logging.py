"""
Logging configuration.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class PtychiEvolveFilter(logging.Filter):
    """Filter that only allows logs from ptychi_evolve package."""
    
    def filter(self, record):
        # Allow logs from ptychi_evolve package or root logger messages that contain ptychi_evolve tags
        return (record.name.startswith('ptychi_evolve') or 
                record.name == '__main__' or
                record.name == 'root' and any(tag in record.getMessage() for tag in [
                    '[INIT]', '[DISCOVERY]', '[GENERATE]', '[EVAL]', '[LLM]', 
                    '[PROCESS]', '[TUNE]', '[EVOLVE]', '[CHECKPOINT]', '[CORRECT]', 
                    '[DECISION]'
                ]))


class CleanFormatter(logging.Formatter):
    """A cleaner formatter that reduces clutter."""
    
    LEVEL_COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Tags that should be highlighted
    HIGHLIGHT_TAGS = {
        '[INIT]': '\033[96m',       # Bright Cyan
        '[DISCOVERY]': '\033[92m',  # Bright Green
        '[GENERATE]': '\033[94m',   # Bright Blue
        '[EVAL]': '\033[93m',       # Bright Yellow
        '[LLM]': '\033[95m',        # Bright Magenta
        '[PROCESS]': '\033[92m',    # Bright Green
        '[TUNE]': '\033[94m',       # Bright Blue
        '[EVOLVE]': '\033[93m',     # Bright Yellow
        '[CHECKPOINT]': '\033[96m', # Bright Cyan
        '[DEBUG]': '\033[36m',      # Cyan
        '[CORRECT]': '\033[33m',    # Yellow
        '[DECISION]': '\033[95m',   # Bright Magenta
    }
    
    def __init__(self, fmt=None, datefmt=None, use_colors=True, show_time=True, 
                 show_logger_name=False, compact=True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()
        self.show_time = show_time
        self.show_logger_name = show_logger_name
        self.compact = compact
        
    def format(self, record):
        # Build the format dynamically based on settings
        parts = []
        
        # Time (simplified)
        if self.show_time:
            if self.compact:
                # Show only time, not date
                time_str = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
            else:
                time_str = self.formatTime(record, self.datefmt)
            
            if self.use_colors:
                parts.append(f"{self.DIM}{time_str}{self.RESET}")
            else:
                parts.append(time_str)
        
        # Logger name (optional, usually not needed for discovery)
        if self.show_logger_name and record.name != 'root':
            short_name = record.name.split('.')[-1]  # Just the last part
            if self.use_colors:
                parts.append(f"{self.DIM}{short_name}{self.RESET}")
            else:
                parts.append(short_name)
        
        # Message
        msg = record.getMessage()
        
        # Apply colors to special tags
        if self.use_colors:
            for tag, color in self.HIGHLIGHT_TAGS.items():
                if tag in msg:
                    msg = msg.replace(tag, f"{color}{self.BOLD}{tag}{self.RESET}")
        
        parts.append(msg)
        
        return ' '.join(parts)


class SingleStreamLogger:
    """
    A logger that ensures messages are only output once, either to file or console.
    This eliminates the duplicate output issue.
    """
    
    def __init__(self, name: str, verbose: bool = False, debug: bool = False):
        self.name = name
        self.verbose = verbose
        self.debug_mode = debug  # Renamed to avoid conflict with debug() method
        self._logger = None
        self._configured = False
        
    def _ensure_configured(self):
        """Ensure the logger is properly configured."""
        if not self._configured:
            self._logger = logging.getLogger(self.name)
            # Set logger level based on verbosity settings
            if self.debug_mode:
                self._logger.setLevel(logging.DEBUG)
            elif self.verbose:
                self._logger.setLevel(logging.INFO)
            else:
                self._logger.setLevel(logging.WARNING)
            self._configured = True
    

    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method that handles both file and console output."""
        self._ensure_configured()
        
        # Always log to file at appropriate level
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        self._logger.log(numeric_level, message, **kwargs)
    
    # Convenience methods for different types of messages
    def init(self, message: str):
        self._log('INFO', f"[INIT] {message}")
        
    def discovery(self, message: str):
        self._log('INFO', f"[DISCOVERY] {message}")
        
    def generate(self, message: str):
        self._log('INFO', f"[GENERATE] {message}")
        
    def eval(self, message: str):
        self._log('INFO', f"[EVAL] {message}")
        
    def llm(self, message: str):
        self._log('INFO', f"[LLM] {message}")
        
    def process(self, message: str, success: bool = True):
        level = 'INFO' if success else 'WARNING'
        symbol = '✓' if success else '✗'
        self._log(level, f"[PROCESS] {symbol} {message}")
        
    def tune(self, message: str):
        self._log('INFO', f"[TUNE] {message}")
        
    def evolve(self, message: str):
        self._log('INFO', f"[EVOLVE] {message}")
        
    def checkpoint(self, message: str):
        self._log('INFO', f"[CHECKPOINT] {message}")
        
    def decision(self, message: str):
        self._log('INFO', f"[DECISION] {message}")
        
    def correct(self, message: str):
        self._log('INFO', f"[CORRECT] {message}")
        
    def info(self, message: str):
        self._log('INFO', message)
        
    def debug(self, message: str):
        self._log('DEBUG', message)
        
    def warning(self, message: str):
        self._log('WARNING', message)
        
    def error(self, message: str, exc_info: bool = False):
        self._log('ERROR', message, exc_info=exc_info)
        

    def prompt(self, message: str):
        """For user prompts."""
        print(message)
        self._log('INFO', f"[PROMPT] {message}")
        
    def debug_info(self, message: str, indent: int = 0):
        """Debug information with optional indentation."""
        prefix = "  " * indent
        self._log('DEBUG', f"{prefix}[DEBUG] {message}")


def setup_logging(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    console_level: str = "INFO",
    use_colors: bool = True,
    compact: bool = True,
    show_time: bool = True,
    show_logger_name: bool = False
) -> None:
    """
    Configure logging.
    
    Args:
        log_file: Path to log file. If None, only console logging is used.
        log_level: Logging level for file output
        console_level: Logging level for console output
        use_colors: Whether to use colors in console output
        compact: Use compact formatting
        show_time: Show timestamps
        show_logger_name: Show logger names (usually not needed)
    """
    # Get root logger and clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # File handler with detailed format
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Add filter to only log ptychi_evolve related messages
        file_handler.addFilter(PtychiEvolveFilter())
        
        # Detailed format for file
        file_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        file_handler.setFormatter(logging.Formatter(file_format))
        root_logger.addHandler(file_handler)
    
    # Console handler with clean format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper()))
    
    # Clean format for console
    console_formatter = CleanFormatter(
        use_colors=use_colors,
        show_time=show_time,
        show_logger_name=show_logger_name,
        compact=compact
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Suppress noisy loggers
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    
    # Initial log message (only to file)
    if log_file:
        logging.getLogger(__name__).info(f"Logging initialized. Writing to {log_file}")


def get_logger(name: str, verbose: bool = False, debug: bool = False) -> SingleStreamLogger:
    """
    Get a logger instance that avoids duplicate output.
    
    Args:
        name: Logger name
        verbose: Enable verbose output
        debug: Enable debug output
        
    Returns:
        SingleStreamLogger instance
    """
    return SingleStreamLogger(name, verbose, debug)