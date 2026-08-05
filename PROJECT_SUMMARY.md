# Project Summary

This repository contains a trained two-stage XGBoost lightning cascade and a
tested real-time serving path using Apache Kafka, Redis, H3, Python, and Docker.
The local milestone is complete; AWS deployment is intentionally not performed.

Verified artifacts:

- `benchmark-results.json`: sub-100 ms local ingestion-to-prediction latency
- `data/models/cascade_efficiency_15min.json`: 98.1803% reduction in stage-two calls
- `data/models/two_stage_evaluation_15min.json`: held-out prediction metrics
