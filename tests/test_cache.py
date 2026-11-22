import time
from app.cache import ResponseCache

class TestResponseCache:
    """Test the response cache."""
    
    def setup_method(self):
        # 1 second TTL for fast testing
        self.cache = ResponseCache(ttl_seconds=1)

    def test_cache_miss_returns_none(self):
        assert self.cache.get("unknown query") is None

    def test_cache_hit_returns_response(self):
        self.cache.set("What is Python?", "A programming language.")
        assert self.cache.get("What is Python?") == "A programming language."

    def test_ttl_expiration(self):
        self.cache.set("query", "response")
        
        # Should hit immediately
        assert self.cache.get("query") == "response"
        
        # Wait for TTL to expire
        time.sleep(1.5)
        
        # Should miss and be removed
        assert self.cache.get("query") is None

    def test_stats_tracking(self):
        # 2 Misses
        self.cache.get("miss1")
        self.cache.get("miss2")
        
        # 1 Hit
        self.cache.set("hit", "value")
        self.cache.get("hit")
        
        stats = self.cache.stats
        
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["cached_entries"] == 1