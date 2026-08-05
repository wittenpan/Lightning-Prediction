# System architecture

## Short answer

Kafka is the durable event bus. Redis is the prediction service's low-latency
working memory. The online path has two long-running application services—not a
prediction service and a training service:

| Service | Reads from | Writes to | Job |
|---|---|---|---|
| Live producer | Blitzortung websocket | Kafka `lightning-strikes` | Decode, validate, normalize, and publish strikes |
| Prediction service | Kafka `lightning-strikes` | Redis and Kafka `lightning-predictions` | Build features, run the cascade, and emit predictions |

Training is a separate offline batch workflow over historical Parquet data. It
produces the committed XGBoost model and metadata files in `data/models/`.
Neither Kafka nor a live event starts training.

## One event, end to end

```text
Blitzortung
    |
    | websocket message
    v
live-producer
    |
    | normalized event keyed by event_id
    v
Kafka topic: lightning-strikes
    |
    | consumer group: lightning-prediction-group
    v
prediction-service
    |-- map latitude/longitude to an H3 cell
    |-- add the event to that cell's Redis sorted set
    |-- query 5/15/30/60/90-minute and neighboring-cell counts
    |-- run XGBoost stage 1
    |      `-- if low signal: skip stage 2
    |-- otherwise run XGBoost stage 2
    |-- cache the latest per-cell prediction in Redis
    v
Kafka topic: lightning-predictions
    |
    `-- benchmark, alerting, API, dashboard, or storage consumer
```

## Why Kafka is there

Kafka separates ingestion from inference. The producer can continue publishing
while the predictor briefly restarts; events remain in the log and a consumer
group resumes from its offsets. Event keys preserve per-key ordering, topics can
be replayed, and output consumers can be added without changing the producer or
prediction service.

The prediction service is both a Kafka consumer and a Kafka producer. With more
input partitions, several prediction-service replicas in the same consumer group
can divide the input workload. Kafka is not used for random access to recent
spatial history—that is Redis's role.

## Why Redis is there

Each prediction needs recent counts for one H3 cell and its ring-1/ring-2
neighbors across several time windows. Re-scanning a Kafka log for those counts
on every event would be slow and awkward. Redis provides:

- sorted sets keyed by H3 cell, scored by strike time;
- pipelined range-count queries for temporal and neighboring-cell features;
- cached static H3 neighbor sets;
- short-lived latest-prediction entries; and
- TTL-based expiry so old online state is removed automatically.

Kafka is therefore the history of events in motion; Redis is the queryable
materialized state needed to make the next prediction quickly.

## Where training fits

The offline ML path is intentionally independent:

```text
historical raw data -> feature/label Parquet splits
  -> train stage 1 (any future activity?)
  -> train stage 2 (meaningful activity, only for stage-1 positives)
  -> tune thresholds and evaluate held-out data
  -> versioned model + metadata artifacts
  -> Docker image used by prediction-service
```

This separation avoids training/serving interference and makes a deployment
reproducible. A future scheduled retraining job could publish a newly validated
model artifact, but that is not part of the current live Kafka flow.

## Local and AWS mapping

| Local container | Managed AWS proof of concept |
|---|---|
| Apache Kafka | Amazon MSK |
| Redis | Amazon ElastiCache for Redis |
| Producer + prediction containers | Docker on one EC2 instance |
| Docker network | VPC and security groups |

The exact same Python services and model artifacts run in both environments;
only connection settings and TLS change.
