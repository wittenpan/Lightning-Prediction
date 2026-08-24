# Real-Time Lightning Prediction System

A local, AWS-ready event pipeline that consumes live Blitzortung lightning data,
normalizes and publishes it to Kafka, builds geospatial/time-window features in
Redis, runs a gated two-stage XGBoost cascade, and publishes predictions back to
Kafka.

Kafka connects the live producer and prediction service; it does **not** trigger
training. Redis is queryable, expiring feature state for the prediction service,
not another event queue. See [ARCHITECTURE.md](ARCHITECTURE.md) for the complete
service and data flow.

```text
Blitzortung websocket
  -> LZW decode + timestamp normalization
  -> Kafka: lightning-strikes
  -> H3 mapping + Redis sorted-set feature windows
  -> XGBoost stage 1 (activity gate)
       -> negative: skip stage 2
       -> positive: XGBoost stage 2 (intensity)
  -> Kafka: lightning-predictions
```

## Verified results

- **Latency:** 1,000 events at 100 events/second from the containerized benchmark:
  8.212 ms p50, **18.483 ms p95**, and 148.299 ms p99 from ingestion through
  prediction. Kafka input-to-output round trip was 50.952 ms p95. See
  [`benchmark-results.json`](benchmark-results.json).
- **Cascade savings:** on 7,331,155 held-out temporal test events, stage one
  routed 133,402 events to stage two and skipped 7,197,753, reducing expensive
  stage-two model invocations by **98.1803%**. See
  [`cascade_efficiency_15min.json`](data/models/cascade_efficiency_15min.json).
- **Model evaluation:** combined precision 0.6956 and F1 0.4306 on the committed
  test artifact. This is an engineering/ML portfolio system, not a safety-grade
  weather warning product.

## Run locally

Prerequisites: Docker Desktop and Docker Compose v2.

```bash
docker compose up -d --build
docker compose ps
```

This starts Apache Kafka 3.9 in KRaft mode, Redis 7, and the containerized
prediction service. Optional inspection tools are available with:

```bash
docker compose --profile tools up -d
```

Generate synthetic traffic from the host:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-streaming.txt
python -m src.streaming.producer --simulate 100 --rate 50
```

Connect the producer to the public live feed by omitting `--simulate`:

```bash
python -m src.streaming.producer
```

The producer handles Blitzortung's LZW websocket messages and rotates through a
reconnecting stream loop.

## Reproduce the claims

```bash
# Unit/integration tests
python -m pytest -q

# Held-out cascade calculation
python -m src.ml.verify_cascade_savings

# Kafka -> Redis -> XGBoost -> Kafka benchmark
python -m src.streaming.benchmark --events 1000 --rate 100 --timeout 30
```

The benchmark exits non-zero if predictions are lost or p95
ingestion-to-prediction latency reaches 100 ms.

## H3 operations dashboard

The private StormSignal dashboard maps recent predictions as H3 cells and shows
the event rate, active-cell count, cascade skip rate, latency, and stage
probabilities. Run the local API and dashboard with:

```bash
docker compose --profile dashboard up -d --build
cd dashboard && npm install && npm run dev
```

Open `http://localhost:3000`, select **Live AWS**, and the browser will read the
loopback-only API at `http://127.0.0.1:8765`. Preview mode is self-contained.

## AWS proof of concept

No AWS resources are created by the local workflow. A guarded CloudFormation
stack and preflight/deploy/destroy helper are documented in
[`infra/aws-poc/README.md`](infra/aws-poc/README.md). The runtime reads all
connection information from environment variables in [`.env.example`](.env.example):

- `KAFKA_BOOTSTRAP_SERVERS` maps to an MSK bootstrap endpoint.
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`, and `REDIS_SSL` map to
  ElastiCache.
- The same Docker image runs on EC2 with the committed model artifacts.
- Kafka TLS/SASL options are supported without source-code substitution.
- `self-hosted` mode puts Kafka and Redis on EC2 for a lower-cost smoke test.
- `managed` mode uses MSK and ElastiCache for a short, explicitly cost-confirmed
  résumé proof; it is not assumed to be universally free.

## Key implementation details

- Kafka-native 5 ms bounded batching instead of a one-second application buffer.
- Nanosecond/millisecond/second timestamps normalized to microseconds at ingress.
- Redis pipelines collapse temporal and neighboring-cell count reads into one
  network round trip.
- Event IDs make Redis strike insertion idempotent while preserving simultaneous
  strikes.
- Serving feature order and tuned thresholds load from model metadata, preventing
  training/serving skew.
