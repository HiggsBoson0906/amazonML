import time
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

class ExperimentLogger:
    """Structured logger for experiment execution and ablation metrics."""
    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path
        self.start_time = time.time()
        
    def get_memory_usage_mb(self) -> float:
        if HAS_PSUTIL:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        return 0.0

    def log(self, message: str):
        elapsed = time.time() - self.start_time
        mem = self.get_memory_usage_mb()
        mem_str = f" [RAM: {mem:.1f}MB]" if mem > 0 else ""
        print(f"[{elapsed:6.1f}s]{mem_str} {message}", flush=True)

class Timer:
    """Context manager for fine-grained section timing."""
    def __init__(self, name: str, logger: Optional[ExperimentLogger] = None):
        self.name = name
        self.logger = logger
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start
        msg = f"Completed {self.name} in {self.elapsed:.2f}s"
        if self.logger:
            self.logger.log(msg)
        else:
            print(f"  {msg}", flush=True)
