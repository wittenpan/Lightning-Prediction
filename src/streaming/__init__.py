"""
Real-time Lightning Prediction Streaming Pipeline

This module implements a Kafka-based streaming architecture for real-time
lightning strike prediction with Redis caching.

Components:
- Producer: Ingests lightning strike data from Blitzortung WebSocket → Kafka
- Consumer: Processes strikes, generates predictions using trained models
- Cache: Redis for H3 cell feature caching and prediction results
"""

__version__ = "1.0.0"
