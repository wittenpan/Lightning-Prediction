# Streaming Pipeline

See [README.md](README.md) for architecture, local orchestration, tests,
benchmarks, and environment variables. The online implementation lives in
`src/streaming/`:

- `producer.py`: live LZW websocket ingestion and synthetic traffic
- `consumer.py`: Kafka/Redis feature path and prediction publishing
- `models.py`: metadata-ordered two-stage XGBoost cascade
- `redis_cache.py`: pipelined time-series and prediction cache
- `benchmark.py`: reproducible end-to-end latency measurement
