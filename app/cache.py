import hashlib
import time
from typing import Optional
#Claude to remeber:
#In production we use Redis for persistance and sharing with multiple instances
#USer Permitheous for dictionary counter and load balancer in front:
class ResponseCache:
    """
    In-memory response cache with TTL (time-to-live).

    In production, replace this with Redis for:
    - Persistence across restarts
    - Shared cache across multiple instances
    - Built-in TTL management
    """

    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self._cache: dict[str, dict] = {}
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str) -> str:
        """Create a cache key from the normalized query."""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        """
        Get cached response if it exists and hasn't expired.
        Returns None on cache miss.
        """
        key = self._make_key(query)

        if key in self._cache:
            entry = self._cache[key]
            # Check TTL
            if time.time() - entry["timestamp"] < self.ttl:
                self._hits += 1
                return entry["response"]
            else:
                # Expired - remove it
                del self._cache[key]
        
        self._misses += 1
        return None

    def set(self, query: str, response: str) -> None:
        """Store a response in the cache with the current timestamp."""
        key = self._make_key(query)
        self._cache[key] = {
            "response": response,
            "timestamp": time.time()
        }

    @property
    def stats(self) -> dict:
        """Cache performance statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': f"{hit_rate:.1%}",
            'cached_entries': len(self._cache),
        }


# ==========================================
# TEST SUITE
# ==========================================
if __name__ == "__main__":
    print("=== TEST 1: Basic Set and Get ===")
    cache = ResponseCache(ttl_seconds=2) # 2 seconds TTL for testing
    
    query = "What is the capital of France?"
    cache.set(query, "The capital of France is Paris.")
    
    # Should be a hit
    print(f"Result: {cache.get(query)}")
    print(f"Stats:  {cache.stats}\n")

    print("=== TEST 2: Normalization (Case & Whitespace) ===")
    # Even with weird casing and spaces, it should hit the same cache key
    messy_query = "   WHAT IS THE CAPITAL OF FRANCE?   "
    print(f"Result: {cache.get(messy_query)}")
    print(f"Stats:  {cache.stats}\n")

    print("=== TEST 3: Cache Miss ===")
    print(f"Result: {cache.get('What is the capital of Germany?')}")
    print(f"Stats:  {cache.stats}\n")

    print("=== TEST 4: TTL Expiration ===")
    print("Sleeping for 2.5 seconds to let the cache expire...")
    time.sleep(2.5)
    
    # Should be a miss now because TTL was 2 seconds
    print(f"Result: {cache.get(query)}") 
    print(f"Stats:  {cache.stats}\n")