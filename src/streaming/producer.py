"""Live/simulated lightning ingestion into Kafka."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import websockets
from kafka import KafkaProducer

try:
    from .schema import decode_blitzortung_message, make_event
    from .settings import Settings
except ImportError:  # Support `python src/streaming/producer.py`.
    from schema import decode_blitzortung_message, make_event
    from settings import Settings

logger = logging.getLogger(__name__)


class LightningStrikeProducer:
    """Decode Blitzortung events and publish the normalized wire schema."""

    def __init__(
        self,
        kafka_bootstrap_servers: str = "localhost:9092",
        kafka_topic: str = "lightning-strikes",
        ws_url: str = "wss://ws2.blitzortung.org/",
        kafka_options: Optional[Dict[str, Any]] = None,
        producer: Optional[KafkaProducer] = None,
    ) -> None:
        self.kafka_topic = kafka_topic
        self.ws_url = ws_url
        self.producer = producer or KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
            key_serializer=lambda key: key.encode("utf-8") if key else None,
            compression_type="gzip",
            batch_size=16_384,
            linger_ms=5,
            acks="all",
            retries=5,
            **(kafka_options or {}),
        )
        self.strikes_published = 0
        self.errors = 0
        self.started_at = time.monotonic()

    def parse_strike(self, message: str) -> Optional[Dict[str, Any]]:
        data = decode_blitzortung_message(message)
        if data is None:
            self.errors += 1
            logger.warning("Unable to decode provider message")
            return None
        try:
            return make_event(data)
        except (TypeError, ValueError) as exc:
            self.errors += 1
            logger.warning("Invalid strike: %s", exc)
            return None

    def publish_strike(self, strike: Dict[str, Any]) -> None:
        """Send immediately; Kafka's native 5 ms linger performs bounded batching."""
        future = self.producer.send(
            self.kafka_topic,
            key=strike["event_id"],
            value=strike,
        )
        future.add_callback(lambda _metadata: self._record_success())
        future.add_errback(self._record_error)

    def _record_success(self) -> None:
        self.strikes_published += 1

    def _record_error(self, exc: BaseException) -> None:
        self.errors += 1
        logger.error("Kafka send failed: %s", exc)

    async def stream(self) -> None:
        reconnect_delay = 1
        while True:
            try:
                logger.info("Connecting to %s", self.ws_url)
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=10_000_000,
                ) as websocket:
                    await websocket.send('{"a":111}')
                    reconnect_delay = 1
                    async for message in websocket:
                        strike = self.parse_strike(message)
                        if strike:
                            self.publish_strike(strike)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Stream disconnected: %s", exc)
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)

    def publish_simulated(self, count: int, rate: float) -> None:
        """Publish deterministic-shape local traffic without contacting Blitzortung."""
        interval = 1.0 / rate if rate > 0 else 0.0
        for _ in range(count):
            now_ns = time.time_ns()
            event = make_event(
                {
                    "lat": random.uniform(27.0, 31.0),
                    "lon": random.uniform(-84.0, -80.0),
                    "time": now_ns,
                    "alt": random.randint(0, 1000),
                    "pol": random.choice((-1, 1)),
                    "mds": random.randint(1, 1_000_000),
                },
                source="simulated",
            )
            self.publish_strike(event)
            if interval:
                time.sleep(interval)
        self.producer.flush()

    def close(self) -> None:
        self.producer.flush()
        self.producer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream lightning events into Kafka")
    parser.add_argument("--simulate", type=int, metavar="COUNT", help="publish COUNT synthetic events")
    parser.add_argument("--rate", type=float, default=100.0, help="synthetic events per second")
    parser.add_argument("--ws-url", default="wss://ws2.blitzortung.org/")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    settings = Settings.from_env()
    producer = LightningStrikeProducer(
        kafka_bootstrap_servers=settings.kafka_bootstrap_servers,
        kafka_topic=settings.input_topic,
        ws_url=args.ws_url,
        kafka_options=settings.kafka_kwargs(),
    )
    logger.info(
        "Producer ready: %s -> %s at %s",
        args.ws_url if args.simulate is None else "simulator",
        settings.input_topic,
        datetime.now(timezone.utc).isoformat(),
    )
    try:
        if args.simulate is not None:
            producer.publish_simulated(args.simulate, args.rate)
        else:
            try:
                asyncio.run(producer.stream())
            except KeyboardInterrupt:
                logger.info("Producer stopped")
    finally:
        producer.close()


if __name__ == "__main__":
    main()
