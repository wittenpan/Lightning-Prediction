"""Publish low-signal H3 candidates so the cascade gate is exercised online."""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any, Optional

import h3
import redis
from kafka import KafkaProducer

from .schema import make_event
from .settings import Settings

logger = logging.getLogger(__name__)


class CandidateCellProducer:
    """Turn recent strike cells and their neighbors into score-only events.

    The lightning feed naturally contains only positive observations. Scoring a
    small neighborhood around recent strikes supplies the no-strike candidates
    that stage one was trained to reject, without recording synthetic strikes.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        seed_limit: int = 24,
        ring: int = 1,
        redis_client: Optional[redis.Redis] = None,
        producer: Optional[KafkaProducer] = None,
    ) -> None:
        self.settings = settings
        self.seed_limit = seed_limit
        self.ring = ring
        self.redis = redis_client or redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            ssl=settings.redis_ssl,
            decode_responses=False,
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        self.producer = producer or KafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
            key_serializer=lambda key: key.encode("utf-8"),
            compression_type="gzip",
            linger_ms=5,
            acks="all",
            retries=5,
            **settings.kafka_kwargs(),
        )

    def recent_seed_cells(self, now_us: int) -> list[str]:
        """Return the busiest recently observed cells, bounded for the POC."""
        keys = list(self.redis.scan_iter(match="strikes:*", count=500))[:1_000]
        if not keys:
            return []
        pipeline = self.redis.pipeline(transaction=False)
        for key in keys:
            pipeline.zcount(key, now_us - 300_000_000, now_us)
        counts = pipeline.execute()
        ranked: list[tuple[int, str]] = []
        for key, count in zip(keys, counts):
            if not count:
                continue
            text = key.decode("utf-8") if isinstance(key, bytes) else str(key)
            ranked.append((int(count), text.split(":", 1)[1]))
        ranked.sort(reverse=True)
        return [cell for _count, cell in ranked[: self.seed_limit]]

    def candidate_cells(self, now_us: int) -> list[str]:
        candidates: set[str] = set()
        for cell in self.recent_seed_cells(now_us):
            candidates.update(h3.grid_disk(cell, self.ring))
        return sorted(candidates)

    def publish_cycle(self) -> int:
        now_ns = time.time_ns()
        cells = self.candidate_cells(now_ns // 1_000)
        for cell in cells:
            latitude, longitude = h3.cell_to_latlng(cell)
            event = make_event(
                {
                    "lat": latitude,
                    "lon": longitude,
                    "time": now_ns,
                    "mds": f"candidate:{cell}:{now_ns}",
                },
                source="candidate-grid",
            )
            event["event_type"] = "candidate"
            event["h3_cell"] = cell
            self.producer.send(self.settings.input_topic, key=event["event_id"], value=event)
        self.producer.flush()
        return len(cells)

    def close(self) -> None:
        self.producer.flush()
        self.producer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score H3 cells around live strikes")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--seed-limit", type=int, default=24)
    parser.add_argument("--ring", type=int, default=1)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    service = CandidateCellProducer(
        Settings.from_env(),
        seed_limit=args.seed_limit,
        ring=args.ring,
    )
    try:
        while True:
            published = service.publish_cycle()
            logger.info("published %d candidate H3 cells", published)
            if args.once:
                break
            time.sleep(max(5.0, args.interval))
    finally:
        service.close()


if __name__ == "__main__":
    main()
