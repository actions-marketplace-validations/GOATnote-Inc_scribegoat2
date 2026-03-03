"""
Enterprise Reliability Module for ScribeGoat2

Provides fault tolerance and reliability patterns:
- Circuit breaker pattern
- Retry with exponential backoff
- Bulkhead pattern for isolation
- Timeout handling
- Graceful degradation

Designed for mission-critical healthcare AI systems
"""

from .bulkhead import (
    Bulkhead,
    BulkheadError,
    with_bulkhead,
)
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    circuit_breaker,
)
from .fallback import (
    FallbackPolicy,
    with_fallback,
)
from .retry import (
    RetryError,
    RetryPolicy,
    retry,
    with_retry,
)
from .timeout import (
    TimeoutError,
    TimeoutPolicy,
    with_timeout,
)

__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerError",
    "circuit_breaker",
    # Retry
    "RetryPolicy",
    "RetryError",
    "retry",
    "with_retry",
    # Timeout
    "TimeoutPolicy",
    "TimeoutError",
    "with_timeout",
    # Bulkhead
    "Bulkhead",
    "BulkheadError",
    "with_bulkhead",
    # Fallback
    "FallbackPolicy",
    "with_fallback",
]

__version__ = "1.0.0"
