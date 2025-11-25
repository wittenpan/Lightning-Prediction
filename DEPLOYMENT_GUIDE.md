# Complete Deployment Guide

Step-by-step guide to deploy the Lightning Prediction System from scratch.

---

## Prerequisites

- **Hardware:** 16GB RAM minimum (tested on M3 Mac)
- **Software:**
  - Python 3.10+ (tested on Python 3.13)
  - Docker Desktop (for Kafka + Redis)
  - Git (for version control)
- **Time:** ~30 minutes for full setup

---

## Part 1: Project Setup (5 minutes)

### Step 1: Clone Repository

```bash
cd ~/Desktop
git clone <your-repo-url> Lightning-Project
cd Lightning-Project
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Verify activation
which python
# Should show: /Users/wittenpan/Desktop/Lightning-Project/.venv/bin/python
```

### Step 3: Install ML Dependencies

```bash
# Install core ML dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, xgboost, h3; print('✓ ML dependencies OK')"
```

**Expected output:**
```
✓ ML dependencies OK
```

### Step 4: Verify Project Structure

```bash
# Check all required files exist
ls -la src/ml/
ls -la src/processing/
ls -la src/streaming/
ls -la data/models/
ls -la configs/

# Should see:
# - src/ml/*.py (training scripts)
# - src/processing/*.py (data pipeline)
# - src/streaming/*.py (Kafka/Redis code)
# - data/models/*.ubj (trained models)
# - configs/grid_config.yml
```

---

## Part 2: Train ML Models (15 minutes)

### Step 5: Check If Models Exist

```bash
ls -lh data/models/stage*.ubj

# If you see:
# stage1_model_15min.ubj
# stage2_model_15min.ubj
# → Skip to Part 3 (models already trained!)

# If missing → Continue with training
```

### Step 6: Prepare Data (if needed)

```bash
# Only if data/processed/ is empty
python -m src.processing.data_preparation

# This will:
# 1. Load raw lightning data
# 2. Create H3 grid
# 3. Engineer features
# 4. Create train/val/test splits
#
# Expected time: ~5 minutes
# Expected output: data/processed/stage1_features_*.parquet
```

### Step 7: Train Models

```bash
# Train Stage 1 (Activity Detection)
python -m src.ml.train_stage1 --horizon 15min

# Expected output:
# Test Performance:
#   ROC-AUC:   0.8263
#   Precision: 0.2200
#   Recall:    0.6550
#   F1 Score:  0.3294
#
# Expected time: ~2 minutes

# Train Stage 2 (Intensity Prediction)
python -m src.ml.train_stage2 --horizon 15min

# Expected output:
# Test Performance:
#   ROC-AUC:   0.9406
#   Precision: 0.2015
#   Recall:    0.8301
#   F1 Score:  0.3242
#
# Expected time: ~1 minute
```

### Step 8: Tune Thresholds

```bash
# Optimize classification thresholds
python -m src.ml.tune_threshold --horizon 15min

# Expected output:
# Best threshold (Stage 1): 0.550
# Best threshold (Stage 2): 0.780
#
# Expected time: ~3 minutes
```

### Step 9: Evaluate Two-Stage System

```bash
# Evaluate combined system
python -m src.ml.eval_two_stage --horizon 15min \
  --stage1-threshold 0.55 \
  --stage2-threshold 0.78

# Expected output:
# Combined Two-Stage System:
#   Precision: 0.6956
#   Recall:    0.3119
#   F1:        0.4306
#   ROC-AUC:   0.7836
#
# Expected time: ~30 seconds
```

**✅ Checkpoint:** You should now have:
- `data/models/stage1_model_15min.ubj`
- `data/models/stage2_model_15min.ubj`
- `data/models/tuned_thresholds_15min.json`

---

## Part 3: Setup Streaming Infrastructure (5 minutes)

### Step 10: Install Docker Desktop

```bash
# Check if Docker is installed
docker --version

# If not installed:
# 1. Download from https://www.docker.com/products/docker-desktop
# 2. Install and start Docker Desktop
# 3. Verify: docker ps
```

### Step 11: Start Kafka + Redis

```bash
# Start all services in background
docker-compose up -d

# Expected output:
# Creating lightning-zookeeper ... done
# Creating lightning-kafka ... done
# Creating lightning-redis ... done
# Creating lightning-kafka-ui ... done
# Creating lightning-redis-commander ... done

# Verify services are running
docker-compose ps

# Expected output (all should show "Up"):
# NAME                         STATUS
# lightning-zookeeper         Up
# lightning-kafka             Up
# lightning-redis             Up
# lightning-kafka-ui          Up
# lightning-redis-commander   Up
```

### Step 12: Verify Services

```bash
# Test Kafka
docker exec -it lightning-kafka kafka-topics --list --bootstrap-server localhost:9092

# Test Redis
docker exec -it lightning-redis redis-cli ping
# Expected output: PONG

# Open monitoring UIs
open http://localhost:8080  # Kafka UI
open http://localhost:8081  # Redis Commander
```

### Step 13: Install Streaming Dependencies

```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Install streaming dependencies
pip install -r requirements-streaming.txt

# Verify installation
python -c "import kafka, redis, h3, websockets; print('✓ Streaming dependencies OK')"

# Expected output:
# ✓ Streaming dependencies OK
```

**✅ Checkpoint:** You should now have:
- Docker containers running (Kafka, Redis, monitoring)
- Python streaming dependencies installed
- Monitoring UIs accessible

---

## Part 4: Run Streaming Pipeline (5 minutes)

### Step 14: Test Producer (Optional - simulated data)

Create a test producer script to simulate strikes:

```bash
# Create test script
cat > test_producer.py << 'EOF'
import json
import time
from kafka import KafkaProducer
from datetime import datetime
import random

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Generate test strikes
for i in range(100):
    strike = {
        'latitude': random.uniform(40, 50),
        'longitude': random.uniform(-10, 10),
        'timestamp': int(datetime.utcnow().timestamp() * 1_000_000),
        'altitude': random.randint(0, 1000),
        'polarity': random.choice([-1, 1]),
        'created_at': datetime.utcnow().isoformat(),
        'source': 'test'
    }

    producer.send('lightning-strikes', value=strike)
    print(f"Sent strike {i+1}/100")
    time.sleep(0.1)

producer.flush()
print("✓ Test complete!")
EOF

# Run test
python test_producer.py

# Expected output:
# Sent strike 1/100
# Sent strike 2/100
# ...
# ✓ Test complete!

# Cleanup
rm test_producer.py
```

### Step 15: Run Consumer

```bash
# Terminal 1: Start consumer
python src/streaming/consumer.py

# Expected output:
# ✓ Loaded models from data/models
# ✓ Connected to Redis: localhost:6379
# Consumer initialized: lightning-strikes → lightning-predictions
# Starting Lightning Prediction Consumer...
#
# ==================================================================
# CONSUMER METRICS
# ==================================================================
#   Strikes processed: 100
#   Predictions made: 100
#   Rate: 50.2 strikes/sec
#   Cache hit rate: 15.3%  (low initially, increases over time)
#   Errors: 0
# ==================================================================
```

### Step 16: Run Producer (Live Data - Optional)

**⚠️ Warning:** This connects to real Blitzortung WebSocket!

```bash
# Terminal 2: Start producer (real data)
python src/streaming/producer.py

# Expected output:
# Producer initialized: localhost:9092 → lightning-strikes
# Connecting to wss://ws.blitzortung.org/...
# ✓ Connected to Blitzortung WebSocket
# Flushed batch: 100 strikes
#
# ==================================================================
# PRODUCER METRICS
# ==================================================================
#   Total strikes published: 1,000
#   Rate: 150.3 strikes/sec
#   Errors: 0 (0.00%)
#   Uptime: 7s
# ==================================================================
```

### Step 17: Monitor System

```bash
# Terminal 3: Monitor Kafka topics
docker exec -it lightning-kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic lightning-predictions \
  --from-beginning \
  --max-messages 5

# Expected output:
# {"h3_cell": "871fb467fffffff", "latitude": 51.5074, "longitude": -0.1278, ...}
# {"h3_cell": "871fb467fffffff", "latitude": 51.5075, "longitude": -0.1279, ...}
# ...

# Terminal 4: Monitor Redis
docker exec -it lightning-redis redis-cli

# In Redis CLI:
redis> KEYS strikes:*
redis> KEYS prediction:*
redis> GET "features:871fb467fffffff"
redis> QUIT
```

**✅ Checkpoint:** You should now have:
- Producer streaming strikes to Kafka
- Consumer making predictions in real-time
- Predictions visible in Kafka output topic
- Cache building up in Redis

---

## Part 5: Verification & Testing (2 minutes)

### Step 18: Run End-to-End Test

```bash
# Stop producer/consumer (Ctrl+C in their terminals)

# Run end-to-end test
cat > test_e2e.py << 'EOF'
"""End-to-end test of streaming pipeline."""
import json
import time
from kafka import KafkaProducer, KafkaConsumer
from datetime import datetime

print("Starting end-to-end test...")

# 1. Produce test strike
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

test_strike = {
    'latitude': 51.5074,
    'longitude': -0.1278,
    'timestamp': int(datetime.utcnow().timestamp() * 1_000_000),
    'altitude': 500,
    'polarity': 1,
    'created_at': datetime.utcnow().isoformat(),
    'source': 'test-e2e'
}

producer.send('lightning-strikes', value=test_strike)
producer.flush()
print("✓ Strike published to Kafka")

# 2. Start consumer in background
import subprocess
consumer_proc = subprocess.Popen(
    ['python', 'src/streaming/consumer.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

print("✓ Consumer started")
time.sleep(5)  # Let it process

# 3. Check for prediction in output topic
consumer = KafkaConsumer(
    'lightning-predictions',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    consumer_timeout_ms=10000
)

found = False
for message in consumer:
    prediction = message.value
    if prediction.get('source') == 'test-e2e':
        print("✓ Prediction found!")
        print(f"  H3 Cell: {prediction['h3_cell']}")
        print(f"  Stage 1: {prediction['prediction']['stage1']}")
        print(f"  Stage 2: {prediction['prediction']['stage2']}")
        found = True
        break

# Cleanup
consumer_proc.terminate()
consumer.close()

if found:
    print("\n✅ END-TO-END TEST PASSED!")
else:
    print("\n❌ END-TO-END TEST FAILED - No prediction found")
EOF

python test_e2e.py
rm test_e2e.py
```

### Step 19: Check Metrics

```bash
# View Kafka UI
open http://localhost:8080
# Navigate to: Topics → lightning-strikes → Messages
# Should see incoming strikes

# View Redis Commander
open http://localhost:8081
# Should see:
# - strikes:* keys (strike history)
# - features:* keys (cached features)
# - prediction:* keys (cached predictions)
```

---

## Part 6: Shutdown & Cleanup

### Stop Services

```bash
# Stop producer/consumer
# Press Ctrl+C in their terminals

# Stop Docker services
docker-compose down

# Expected output:
# Stopping lightning-redis-commander ... done
# Stopping lightning-kafka-ui ... done
# Stopping lightning-redis ... done
# Stopping lightning-kafka ... done
# Stopping lightning-zookeeper ... done
# Removing containers...
```

### Clean Data (Optional)

```bash
# Remove all Docker data
docker-compose down -v

# Remove processed data (keeps raw data)
rm -rf data/processed/*

# Remove trained models
rm -rf data/models/*
```

---

## Troubleshooting

### Problem: Docker services won't start

**Solution:**
```bash
# Check Docker Desktop is running
docker ps

# Check port conflicts
lsof -i :9092  # Kafka
lsof -i :6379  # Redis
lsof -i :8080  # Kafka UI

# Kill conflicting processes
kill -9 <PID>

# Restart services
docker-compose down
docker-compose up -d
```

### Problem: Producer can't connect to WebSocket

**Solution:**
```bash
# Test WebSocket connectivity
curl -I https://ws.blitzortung.org

# Check firewall settings
# Use test producer instead (see Step 14)
```

### Problem: Consumer can't load models

**Solution:**
```bash
# Verify models exist
ls -lh data/models/stage*.ubj

# If missing, retrain
python -m src.ml.train_stage1 --horizon 15min
python -m src.ml.train_stage2 --horizon 15min

# Check model paths in consumer
python -c "from pathlib import Path; print(Path('data/models').resolve())"
```

### Problem: Redis connection refused

**Solution:**
```bash
# Check Redis container
docker logs lightning-redis

# Test connection
docker exec -it lightning-redis redis-cli ping

# Restart Redis
docker-compose restart redis
```

### Problem: Kafka consumer lag

**Solution:**
```bash
# Check consumer group lag
docker exec -it lightning-kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --describe \
  --group lightning-prediction-group

# Reset offsets if needed
docker exec -it lightning-kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --group lightning-prediction-group \
  --reset-offsets \
  --to-latest \
  --topic lightning-strikes \
  --execute
```

---

## Production Deployment Checklist

Before deploying to production:

- [ ] **Security:**
  - [ ] Enable Kafka authentication (SASL/SSL)
  - [ ] Set Redis password
  - [ ] Use environment variables for credentials
  - [ ] Enable firewall rules

- [ ] **Monitoring:**
  - [ ] Set up Prometheus metrics
  - [ ] Configure Grafana dashboards
  - [ ] Add alerting (PagerDuty, Slack)
  - [ ] Monitor Kafka lag

- [ ] **Scaling:**
  - [ ] Add more Kafka partitions
  - [ ] Deploy multiple consumers
  - [ ] Set up Redis Sentinel/Cluster
  - [ ] Configure auto-scaling

- [ ] **Reliability:**
  - [ ] Set up backup/restore for models
  - [ ] Configure Kafka retention
  - [ ] Add dead letter queue
  - [ ] Implement circuit breakers

- [ ] **Testing:**
  - [ ] Load testing (1000+ strikes/sec)
  - [ ] Failover testing (kill services)
  - [ ] Data quality validation
  - [ ] A/B testing for new models

---

## Quick Reference

**Start Everything:**
```bash
# 1. Start infrastructure
docker-compose up -d

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Run consumer (Terminal 1)
python src/streaming/consumer.py

# 4. Run producer (Terminal 2)
python src/streaming/producer.py
```

**Stop Everything:**
```bash
# 1. Stop producer/consumer
Ctrl+C in their terminals

# 2. Stop Docker
docker-compose down
```

**View Logs:**
```bash
docker-compose logs -f kafka
docker-compose logs -f redis
```

**Useful Commands:**
```bash
# List Kafka topics
docker exec -it lightning-kafka kafka-topics --list --bootstrap-server localhost:9092

# View Kafka messages
docker exec -it lightning-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic lightning-strikes --from-beginning

# Redis CLI
docker exec -it lightning-redis redis-cli

# Check Redis memory
docker exec -it lightning-redis redis-cli INFO memory
```

---

## Next Steps

1. **Improve Models:** Try ensemble methods, better features
2. **Add Monitoring:** Prometheus + Grafana
3. **Deploy to Cloud:** AWS, GCP, or Azure
4. **Add API:** REST API for predictions
5. **Build Dashboard:** Real-time visualization

---

## Support

- **Documentation:** See `STREAMING_README.md` for detailed API docs
- **Examples:** Check `PROJECT_SUMMARY.md` for use cases
- **Issues:** GitHub Issues or email

Good luck! 🚀
