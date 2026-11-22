#In producction we use Prometheous
#for structured logging we always use Json formator bcz we don't use prints 
import logging
import json
import time
from datetime import datetime, timezone
from typing import Any

# ==========================================
# 1. STRUCTURED JSON LOGGING
# ==========================================
class JSONFormatter(logging.Formatter):
    """Format log records as JSON for log aggregation (ELK, Datadog, etc.)."""

    def format(self, record):
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        
        # Merge any extra data attached to the record
        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)
            
        return json.dumps(log_obj)

def get_logger(name: str = "production-api") -> logging.Logger:
    """Create a structured JSON logger."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
    return logger


# ==========================================
# 2. METRICS COLLECTOR
# ==========================================
class MetricsCollector:
    def __init__(self):
        self._requests_total = 0
        self._errors_total = 0
        self._latency_sum = 0.0
        self._latency_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._tokens_input = 0
        self._tokens_output = 0

    def record_request(self, latency_ms: float, cache_hit: bool = False, error: bool = False, input_tokens: int = 0, output_tokens: int = 0):
        self._requests_total += 1
        self._latency_sum += latency_ms
        self._latency_count += 1
        
        if error:
            self._errors_total += 1
        if cache_hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1
            
        self._tokens_input += input_tokens
        self._tokens_output += output_tokens

    @property
    def summary(self) -> dict:
        avg_latency = self._latency_sum / self._latency_count if self._latency_count > 0 else 0.0
        total_cache = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_cache if total_cache > 0 else 0.0
        error_rate = self._errors_total / self._requests_total if self._requests_total > 0 else 0.0
        
        return {
            "total_requests": self._requests_total,
            "total_errors": self._errors_total,
            "error_rate": f"{error_rate:.1%}",
            "avg_latemmcy_ms": round(avg_latency, 2),
            "total_input_tokens": self._tokens_input,
            "total_output_tokens": self._tokens_output,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": f"{hit_rate:.1%}"
        }


# ==========================================
# 3. REQUEST TIMER UTILITY
# ==========================================
class RequestTimer:
    """Context manager for timing requests."""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        pass # We don't need to do anything here anymore

    @property
    def elapsed_ms(self) -> float:
        """Dynamically calculate elapsed time in milliseconds."""
        return (time.time() - self.start) * 1000

# ==========================================
# TEST SUITE (From Screenshot 7)
# ==========================================
if __name__ == "__main__":
    logger = get_logger()
    metrics = MetricsCollector()

    print('=== STRUCTURED JSON LOGGING ===')
    # Simulating extra data attached to logs
    logger.info('Processing request', extra={'extra_data': {'user_id': 'user-123', 'thread_id': 't-456'}})
    logger.warning('Rate limit approaching', extra={'extra_data': {'current_rate': 18, 'limit': 20}})
    
    print()
    print('=== METRICS COLLECTION ===')
    print()

    # Simulate some requests
    with RequestTimer() as timer:
        time.sleep(0.1)  # Simulate work
    metrics.record_request(latency_ms=timer.elapsed_ms, input_tokens=50, output_tokens=100, cache_hit=False)
    print(f'Request 1: {timer.elapsed_ms:.1f}ms')

    with RequestTimer() as timer:
        time.sleep(0.05)
    metrics.record_request(latency_ms=timer.elapsed_ms, input_tokens=30, output_tokens=80, cache_hit=True)
    print(f'Request 2: {timer.elapsed_ms:.1f}ms (cache hit)')

    # Simulate an error request (from the very bottom of the last screenshot)
    metrics.record_request(latency_ms=5.0, error=True)

    print("\n=== AGGREGATED METRICS ===")
    # Print out the calculated stats
    print(json.dumps(metrics.stats, indent=2))