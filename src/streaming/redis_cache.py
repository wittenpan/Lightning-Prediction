"""
Redis Cache Layer for Lightning Prediction System

Caches:
- H3 cell features (temporal windows: 5min, 15min, 30min, 60min, 90min)
- Neighbor cell data (ring1, ring2)
- Prediction results (Stage 1 & Stage 2)
- Strike history per cell

Uses Redis Sorted Sets for time-series data with automatic expiration.
"""

import json
import logging
import redis
from typing import Dict, List, Optional, Any, Set, Sequence, Tuple
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LightningCache:
    """
    Redis cache for lightning strike features and predictions.

    Key Patterns:
    - strikes:{h3_cell} → Sorted set of strike timestamps
    - features:{h3_cell}:{window} → JSON of computed features
    - prediction:{h3_cell}:stage1 → Stage 1 prediction + confidence
    - prediction:{h3_cell}:stage2 → Stage 2 prediction + confidence
    - neighbors:{h3_cell}:ring{n} → Set of neighbor cells

    TTL Strategy:
    - Strike data: 2 hours (max lookback window + buffer)
    - Features: 10 minutes
    - Predictions: 5 minutes
    - Neighbors: No expiration (static)
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ssl: bool = False,
    ):
        """
        Initialize Redis connection.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Optional password
        """
        self.redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            ssl=ssl,
            decode_responses=False,  # We'll handle encoding
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True
        )

        # TTL constants (seconds)
        self.STRIKE_TTL = 7200  # 2 hours
        self.FEATURE_TTL = 600   # 10 minutes
        self.PREDICTION_TTL = 300  # 5 minutes

        # Test connection
        try:
            self.redis.ping()
            logger.info(f"✓ Connected to Redis: {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"✗ Redis connection failed: {e}")
            raise

    # ===== STRIKE MANAGEMENT =====

    def add_strike(
        self,
        h3_cell: str,
        timestamp_us: int,
        event_id: Optional[str] = None,
    ):
        """
        Add strike to cell's strike history (sorted set by timestamp).

        Args:
            h3_cell: H3 cell ID
            timestamp_us: Strike timestamp (microseconds since epoch)
        """
        key = f"strikes:{h3_cell}"

        # A timestamp is not unique: multiple sensors can report strikes in the
        # same microsecond. Use the event id as member and time as sorted-set score.
        member = event_id or str(timestamp_us)
        pipeline = self.redis.pipeline(transaction=False)
        pipeline.zadd(key, {member: timestamp_us})
        pipeline.expire(key, self.STRIKE_TTL)
        pipeline.execute()

    def get_strikes(
        self,
        h3_cell: str,
        start_time_us: int,
        end_time_us: int
    ) -> List[int]:
        """
        Get strikes in time window for cell.

        Args:
            h3_cell: H3 cell ID
            start_time_us: Window start (microseconds)
            end_time_us: Window end (microseconds)

        Returns:
            List of strike timestamps in window
        """
        key = f"strikes:{h3_cell}"

        # Query sorted set by score range
        strikes = self.redis.zrangebyscore(
            key, start_time_us, end_time_us, withscores=True
        )
        return [int(score) for _member, score in strikes]

    def get_feature_counts(
        self,
        h3_cell: str,
        neighbors_by_ring: Sequence[Tuple[int, Sequence[str]]],
        current_time_us: int,
        windows_us: Dict[str, int],
        neighbor_window_us: int,
    ) -> Tuple[Dict[str, int], Dict[int, List[int]]]:
        """Fetch all temporal and spatial counts in one Redis round trip."""
        pipeline = self.redis.pipeline(transaction=False)
        primary_key = f"strikes:{h3_cell}"
        for window_us in windows_us.values():
            pipeline.zcount(primary_key, current_time_us - window_us, current_time_us)
        for _ring, neighbors in neighbors_by_ring:
            for neighbor in neighbors:
                pipeline.zcount(
                    f"strikes:{neighbor}",
                    current_time_us - neighbor_window_us,
                    current_time_us,
                )
        results = iter(pipeline.execute())
        window_counts = {name: int(next(results)) for name in windows_us}
        neighbor_counts = {
            ring: [int(next(results)) for _neighbor in neighbors]
            for ring, neighbors in neighbors_by_ring
        }
        return window_counts, neighbor_counts

    def get_strike_count(
        self,
        h3_cell: str,
        start_time_us: int,
        end_time_us: int
    ) -> int:
        """
        Get strike count in time window (faster than get_strikes).

        Args:
            h3_cell: H3 cell ID
            start_time_us: Window start (microseconds)
            end_time_us: Window end (microseconds)

        Returns:
            Number of strikes in window
        """
        key = f"strikes:{h3_cell}"
        return self.redis.zcount(key, start_time_us, end_time_us)

    def cleanup_old_strikes(self, h3_cell: str, before_time_us: int):
        """
        Remove strikes older than timestamp (keep Redis lean).

        Args:
            h3_cell: H3 cell ID
            before_time_us: Remove strikes before this time
        """
        key = f"strikes:{h3_cell}"
        removed = self.redis.zremrangebyscore(key, '-inf', before_time_us)
        return removed

    # ===== FEATURE CACHING =====

    def cache_features(
        self,
        h3_cell: str,
        features: Dict[str, float],
        timestamp_s: Optional[int] = None
    ):
        """
        Cache computed features for cell.

        Args:
            h3_cell: H3 cell ID
            features: Feature dictionary
            timestamp_s: Optional timestamp (defaults to now)
        """
        if timestamp_s is None:
            timestamp_s = int(datetime.utcnow().timestamp())

        key = f"features:{h3_cell}"

        # Store as JSON with metadata
        data = {
            'features': features,
            'timestamp': timestamp_s,
            'cached_at': datetime.utcnow().isoformat()
        }

        self.redis.setex(
            key,
            self.FEATURE_TTL,
            json.dumps(data)
        )

    def get_features(
        self,
        h3_cell: str,
        max_age_seconds: int = 60
    ) -> Optional[Dict[str, float]]:
        """
        Get cached features if fresh enough.

        Args:
            h3_cell: H3 cell ID
            max_age_seconds: Max age to consider fresh

        Returns:
            Feature dict or None if stale/missing
        """
        key = f"features:{h3_cell}"
        data_bytes = self.redis.get(key)

        if not data_bytes:
            return None

        try:
            data = json.loads(data_bytes)
            cached_time = data['timestamp']
            now = int(datetime.utcnow().timestamp())

            # Check freshness
            if (now - cached_time) <= max_age_seconds:
                return data['features']
            else:
                return None  # Stale

        except (json.JSONDecodeError, KeyError):
            return None

    # ===== PREDICTION CACHING =====

    def cache_prediction(
        self,
        h3_cell: str,
        stage: int,
        prediction: int,
        probability: float,
        timestamp_s: Optional[int] = None
    ):
        """
        Cache prediction result.

        Args:
            h3_cell: H3 cell ID
            stage: 1 or 2
            prediction: Binary prediction (0 or 1)
            probability: Model confidence [0, 1]
            timestamp_s: Optional timestamp
        """
        if timestamp_s is None:
            timestamp_s = int(datetime.utcnow().timestamp())

        key = f"prediction:{h3_cell}:stage{stage}"

        data = {
            'prediction': prediction,
            'probability': probability,
            'timestamp': timestamp_s,
            'cached_at': datetime.utcnow().isoformat()
        }

        self.redis.setex(
            key,
            self.PREDICTION_TTL,
            json.dumps(data)
        )

    def cache_prediction_pair(
        self,
        h3_cell: str,
        predictions: Dict[str, Dict[str, Any]],
        timestamp_s: int,
    ) -> None:
        """Cache both cascade stages in one network round trip."""
        cached_at = datetime.utcnow().isoformat()
        pipeline = self.redis.pipeline(transaction=False)
        for stage_number, stage_name in ((1, "stage1"), (2, "stage2")):
            result = predictions[stage_name]
            payload = {
                "prediction": int(result["prediction"]),
                "probability": float(result["probability"]),
                "timestamp": timestamp_s,
                "cached_at": cached_at,
            }
            pipeline.setex(
                f"prediction:{h3_cell}:stage{stage_number}",
                self.PREDICTION_TTL,
                json.dumps(payload),
            )
        pipeline.execute()

    def get_prediction(
        self,
        h3_cell: str,
        stage: int,
        max_age_seconds: int = 30
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached prediction if fresh.

        Args:
            h3_cell: H3 cell ID
            stage: 1 or 2
            max_age_seconds: Max age to consider fresh

        Returns:
            Prediction dict or None if stale/missing
        """
        key = f"prediction:{h3_cell}:stage{stage}"
        data_bytes = self.redis.get(key)

        if not data_bytes:
            return None

        try:
            data = json.loads(data_bytes)
            cached_time = data['timestamp']
            now = int(datetime.utcnow().timestamp())

            if (now - cached_time) <= max_age_seconds:
                return data
            else:
                return None

        except (json.JSONDecodeError, KeyError):
            return None

    # ===== NEIGHBOR CACHING =====

    def cache_neighbors(
        self,
        h3_cell: str,
        ring1: Set[str],
        ring2: Set[str]
    ):
        """
        Cache neighbor cells (static, no TTL).

        Args:
            h3_cell: H3 cell ID
            ring1: Set of ring-1 neighbors
            ring2: Set of ring-2 neighbors
        """
        # Store as sets for O(1) membership checks
        self.redis.sadd(f"neighbors:{h3_cell}:ring1", *ring1)
        self.redis.sadd(f"neighbors:{h3_cell}:ring2", *ring2)

    def get_neighbors(
        self,
        h3_cell: str,
        ring: int
    ) -> Set[str]:
        """
        Get cached neighbors.

        Args:
            h3_cell: H3 cell ID
            ring: 1 or 2

        Returns:
            Set of neighbor cell IDs
        """
        key = f"neighbors:{h3_cell}:ring{ring}"
        members = self.redis.smembers(key)
        return {m.decode('utf-8') for m in members}

    # ===== BATCH OPERATIONS =====

    def get_features_batch(
        self,
        h3_cells: List[str],
        max_age_seconds: int = 60
    ) -> Dict[str, Optional[Dict[str, float]]]:
        """
        Get features for multiple cells (optimized).

        Args:
            h3_cells: List of H3 cell IDs
            max_age_seconds: Max age to consider fresh

        Returns:
            Dict mapping cell → features (None if missing)
        """
        if not h3_cells:
            return {}

        # Build pipeline for batch GET
        pipe = self.redis.pipeline()
        for cell in h3_cells:
            pipe.get(f"features:{cell}")

        results = pipe.execute()

        # Parse results
        output = {}
        now = int(datetime.utcnow().timestamp())

        for cell, data_bytes in zip(h3_cells, results):
            if data_bytes:
                try:
                    data = json.loads(data_bytes)
                    if (now - data['timestamp']) <= max_age_seconds:
                        output[cell] = data['features']
                    else:
                        output[cell] = None  # Stale
                except (json.JSONDecodeError, KeyError):
                    output[cell] = None
            else:
                output[cell] = None

        return output

    # ===== METRICS =====

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        info = self.redis.info()

        # Count keys by pattern
        strike_keys = len(list(self.redis.scan_iter("strikes:*")))
        feature_keys = len(list(self.redis.scan_iter("features:*")))
        prediction_keys = len(list(self.redis.scan_iter("prediction:*")))
        neighbor_keys = len(list(self.redis.scan_iter("neighbors:*")))

        return {
            'total_keys': info['db0']['keys'] if 'db0' in info else 0,
            'strike_keys': strike_keys,
            'feature_keys': feature_keys,
            'prediction_keys': prediction_keys,
            'neighbor_keys': neighbor_keys,
            'memory_used_mb': info['used_memory'] / (1024 * 1024),
            'connected_clients': info['connected_clients'],
            'uptime_seconds': info['uptime_in_seconds']
        }

    def clear_all(self):
        """Clear all cache (use with caution!)."""
        self.redis.flushdb()
        logger.warning("All cache cleared!")


# Example usage
if __name__ == "__main__":
    cache = LightningCache()

    # Add strikes
    cell = "871fb467fffffff"
    now_us = int(datetime.utcnow().timestamp() * 1_000_000)

    for i in range(10):
        cache.add_strike(cell, now_us - i * 60 * 1_000_000)  # 1 strike per minute

    # Query strikes in last 5 minutes
    five_min_ago = now_us - 5 * 60 * 1_000_000
    strikes = cache.get_strikes(cell, five_min_ago, now_us)
    print(f"Strikes in last 5 min: {len(strikes)}")

    # Cache features
    features = {
        'strike_count_5min': 5,
        'strike_count_15min': 15,
        'strike_density_5min': 0.97
    }
    cache.cache_features(cell, features)

    # Retrieve features
    cached = cache.get_features(cell, max_age_seconds=60)
    print(f"Cached features: {cached}")

    # Stats
    stats = cache.get_stats()
    print(f"Cache stats: {json.dumps(stats, indent=2)}")
