"""
Kafka Producer: Lightning Strike Data Ingestion

Connects to Blitzortung WebSocket, parses strikes, and publishes to Kafka.
Handles reconnection, batching, and error recovery.
"""

import json
import logging
import asyncio
import websockets
from typing import Optional, Dict, Any
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LightningStrikeProducer:
    """
    Kafka producer for lightning strike data from Blitzortung network.

    Features:
    - WebSocket connection with auto-reconnect
    - Batched Kafka publishing (configurable batch size)
    - JSON serialization with schema validation
    - Metrics tracking (strikes/sec, latency, errors)
    """

    def __init__(
        self,
        kafka_bootstrap_servers: str = 'localhost:9092',
        kafka_topic: str = 'lightning-strikes',
        ws_url: str = 'wss://ws.blitzortung.org/',
        batch_size: int = 100,
        batch_timeout_ms: int = 1000
    ):
        """
        Initialize Kafka producer and WebSocket connection.

        Args:
            kafka_bootstrap_servers: Kafka broker address
            kafka_topic: Topic to publish strikes
            ws_url: Blitzortung WebSocket URL
            batch_size: Number of strikes to batch before sending
            batch_timeout_ms: Max time to wait before flushing batch
        """
        self.kafka_topic = kafka_topic
        self.ws_url = ws_url
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms

        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            # Performance settings
            compression_type='gzip',  # Compress messages
            batch_size=16384,  # 16KB batches
            linger_ms=10,  # Wait 10ms to batch messages
            acks='all',  # Wait for all replicas
            retries=3
        )

        # Metrics
        self.strikes_published = 0
        self.errors = 0
        self.start_time = datetime.now()
        self.batch_buffer = []
        self.last_flush = datetime.now()

        logger.info(f"Producer initialized: {kafka_bootstrap_servers} → {kafka_topic}")

    def parse_strike(self, message: str) -> Optional[Dict[str, Any]]:
        """
        Parse WebSocket message to lightning strike data.

        Message format: {"lat": 51.5, "lon": -0.1, "time": 1234567890, "alt": 100, ...}

        Returns:
            Parsed strike dict or None if invalid
        """
        try:
            data = json.loads(message)

            # Validate required fields
            required = ['lat', 'lon', 'time']
            if not all(k in data for k in required):
                logger.warning(f"Missing required fields: {data}")
                return None

            # Enrich with metadata
            strike = {
                'latitude': float(data['lat']),
                'longitude': float(data['lon']),
                'timestamp': int(data['time']),  # Unix timestamp (microseconds)
                'altitude': int(data.get('alt', 0)),
                'polarity': int(data.get('pol', 0)),
                'created_at': datetime.utcnow().isoformat(),
                'source': 'blitzortung'
            }

            return strike

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Parse error: {e}, message: {message[:100]}")
            self.errors += 1
            return None

    async def publish_strike(self, strike: Dict[str, Any]):
        """
        Publish strike to Kafka (batched).

        Uses H3 cell as partition key for locality.
        """
        try:
            # Use timestamp as key for ordering within partition
            key = str(strike['timestamp'])

            # Add to batch buffer
            self.batch_buffer.append((key, strike))

            # Flush if batch is full or timeout reached
            now = datetime.now()
            time_since_flush = (now - self.last_flush).total_seconds() * 1000

            if len(self.batch_buffer) >= self.batch_size or time_since_flush >= self.batch_timeout_ms:
                await self.flush_batch()

        except Exception as e:
            logger.error(f"Publish error: {e}")
            self.errors += 1

    async def flush_batch(self):
        """Flush current batch to Kafka."""
        if not self.batch_buffer:
            return

        try:
            for key, strike in self.batch_buffer:
                future = self.producer.send(
                    self.kafka_topic,
                    key=key,
                    value=strike
                )
                # Non-blocking callback
                future.add_callback(self._on_send_success)
                future.add_errback(self._on_send_error)

            # Wait for all messages to be sent
            self.producer.flush()

            self.strikes_published += len(self.batch_buffer)
            logger.info(f"Flushed batch: {len(self.batch_buffer)} strikes")

            self.batch_buffer = []
            self.last_flush = datetime.now()

        except KafkaError as e:
            logger.error(f"Kafka error during flush: {e}")
            self.errors += 1

    def _on_send_success(self, metadata):
        """Callback for successful Kafka send."""
        pass  # Success - no action needed

    def _on_send_error(self, exc):
        """Callback for failed Kafka send."""
        logger.error(f"Kafka send failed: {exc}")
        self.errors += 1

    async def stream(self):
        """
        Main streaming loop: WebSocket → Kafka.

        Handles reconnection automatically.
        """
        reconnect_delay = 1
        max_reconnect_delay = 60

        while True:
            try:
                logger.info(f"Connecting to {self.ws_url}...")
                async with websockets.connect(self.ws_url) as websocket:
                    logger.info("✓ Connected to Blitzortung WebSocket")
                    reconnect_delay = 1  # Reset delay on successful connection

                    # Stream messages
                    async for message in websocket:
                        strike = self.parse_strike(message)
                        if strike:
                            await self.publish_strike(strike)

                            # Log metrics every 1000 strikes
                            if self.strikes_published % 1000 == 0:
                                self.log_metrics()

            except websockets.exceptions.WebSocketException as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

            except KeyboardInterrupt:
                logger.info("Shutting down gracefully...")
                await self.flush_batch()
                self.producer.close()
                break

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await asyncio.sleep(reconnect_delay)

    def log_metrics(self):
        """Log performance metrics."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.strikes_published / elapsed if elapsed > 0 else 0
        error_rate = self.errors / self.strikes_published if self.strikes_published > 0 else 0

        logger.info("="*60)
        logger.info("PRODUCER METRICS")
        logger.info("="*60)
        logger.info(f"  Total strikes published: {self.strikes_published:,}")
        logger.info(f"  Rate: {rate:.1f} strikes/sec")
        logger.info(f"  Errors: {self.errors} ({error_rate*100:.2f}%)")
        logger.info(f"  Uptime: {elapsed:.0f}s")
        logger.info("="*60)

    async def run(self):
        """Start streaming."""
        logger.info("Starting Lightning Strike Producer...")
        await self.stream()


async def main():
    """Example usage."""
    producer = LightningStrikeProducer(
        kafka_bootstrap_servers='localhost:9092',
        kafka_topic='lightning-strikes',
        batch_size=100,
        batch_timeout_ms=1000
    )

    await producer.run()


if __name__ == "__main__":
    asyncio.run(main())
