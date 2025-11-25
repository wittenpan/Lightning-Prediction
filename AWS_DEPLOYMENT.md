# AWS Deployment Guide

Deploy the Lightning Prediction System to AWS for testing and production use.

---

## 🎯 Deployment Overview

We'll use AWS services to replicate the local Docker Compose setup:

| Component | AWS Service | Purpose |
|-----------|-------------|---------|
| Kafka | Amazon MSK (Managed Streaming for Kafka) | Event streaming |
| Redis | Amazon ElastiCache (Redis) | Feature caching |
| ML Models | Amazon S3 | Model storage |
| Consumer/Producer | EC2 instances | Streaming pipeline |
| Monitoring | CloudWatch | Logs & metrics |
| VPC | Amazon VPC | Networking |

**Estimated Cost:** ~$50-100/month for testing (can be optimized)

---

## 📋 Prerequisites

1. **AWS Account** with billing enabled
2. **AWS CLI** installed and configured
3. **Terraform** (optional - for infrastructure as code)
4. **SSH Key Pair** for EC2 access

---

## Part 1: AWS Setup (10 minutes)

### Step 1: Install AWS CLI

```bash
# macOS
brew install awscli

# Verify installation
aws --version
# Should show: aws-cli/2.x.x
```

### Step 2: Configure AWS Credentials

```bash
# Configure AWS CLI
aws configure

# Enter your credentials:
# AWS Access Key ID: YOUR_ACCESS_KEY
# AWS Secret Access Key: YOUR_SECRET_KEY
# Default region: us-east-1
# Default output format: json

# Verify configuration
aws sts get-caller-identity
```

### Step 3: Create SSH Key Pair

```bash
# Create key pair in AWS
aws ec2 create-key-pair \
  --key-name lightning-key \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/lightning-key.pem

# Set permissions
chmod 400 ~/.ssh/lightning-key.pem

# Verify
ls -l ~/.ssh/lightning-key.pem
```

---

## Part 2: Network Setup (15 minutes)

### Step 4: Create VPC and Subnets

```bash
# Create VPC
VPC_ID=$(aws ec2 create-vpc \
  --cidr-block 10.0.0.0/16 \
  --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=lightning-vpc}]' \
  --query 'Vpc.VpcId' \
  --output text)

echo "VPC ID: $VPC_ID"

# Enable DNS
aws ec2 modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-hostnames
aws ec2 modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-support

# Create Internet Gateway
IGW_ID=$(aws ec2 create-internet-gateway \
  --tag-specifications 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=lightning-igw}]' \
  --query 'InternetGateway.InternetGatewayId' \
  --output text)

# Attach to VPC
aws ec2 attach-internet-gateway --vpc-id $VPC_ID --internet-gateway-id $IGW_ID

# Create Public Subnet (for EC2)
SUBNET_PUBLIC=$(aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block 10.0.1.0/24 \
  --availability-zone us-east-1a \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=lightning-public}]' \
  --query 'Subnet.SubnetId' \
  --output text)

# Create Private Subnet 1 (for MSK/ElastiCache)
SUBNET_PRIVATE_1=$(aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block 10.0.2.0/24 \
  --availability-zone us-east-1a \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=lightning-private-1}]' \
  --query 'Subnet.SubnetId' \
  --output text)

# Create Private Subnet 2 (for MSK/ElastiCache HA)
SUBNET_PRIVATE_2=$(aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block 10.0.3.0/24 \
  --availability-zone us-east-1b \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=lightning-private-2}]' \
  --query 'Subnet.SubnetId' \
  --output text)

echo "Public Subnet: $SUBNET_PUBLIC"
echo "Private Subnet 1: $SUBNET_PRIVATE_1"
echo "Private Subnet 2: $SUBNET_PRIVATE_2"

# Create Route Table
RTB_ID=$(aws ec2 create-route-table \
  --vpc-id $VPC_ID \
  --tag-specifications 'ResourceType=route-table,Tags=[{Key=Name,Value=lightning-rtb}]' \
  --query 'RouteTable.RouteTableId' \
  --output text)

# Add route to Internet Gateway
aws ec2 create-route \
  --route-table-id $RTB_ID \
  --destination-cidr-block 0.0.0.0/0 \
  --gateway-id $IGW_ID

# Associate with public subnet
aws ec2 associate-route-table \
  --subnet-id $SUBNET_PUBLIC \
  --route-table-id $RTB_ID
```

### Step 5: Create Security Groups

```bash
# Security Group for EC2 (Consumer/Producer)
SG_EC2=$(aws ec2 create-security-group \
  --group-name lightning-ec2-sg \
  --description "Lightning EC2 instances" \
  --vpc-id $VPC_ID \
  --query 'GroupId' \
  --output text)

# Allow SSH
aws ec2 authorize-security-group-ingress \
  --group-id $SG_EC2 \
  --protocol tcp \
  --port 22 \
  --cidr 0.0.0.0/0

# Allow all outbound
aws ec2 authorize-security-group-egress \
  --group-id $SG_EC2 \
  --protocol all \
  --cidr 0.0.0.0/0

# Security Group for MSK (Kafka)
SG_MSK=$(aws ec2 create-security-group \
  --group-name lightning-msk-sg \
  --description "Lightning MSK cluster" \
  --vpc-id $VPC_ID \
  --query 'GroupId' \
  --output text)

# Allow Kafka from EC2
aws ec2 authorize-security-group-ingress \
  --group-id $SG_MSK \
  --protocol tcp \
  --port 9092 \
  --source-group $SG_EC2

# Security Group for ElastiCache (Redis)
SG_REDIS=$(aws ec2 create-security-group \
  --group-name lightning-redis-sg \
  --description "Lightning Redis cluster" \
  --vpc-id $VPC_ID \
  --query 'GroupId' \
  --output text)

# Allow Redis from EC2
aws ec2 authorize-security-group-ingress \
  --group-id $SG_REDIS \
  --protocol tcp \
  --port 6379 \
  --source-group $SG_EC2

echo "EC2 Security Group: $SG_EC2"
echo "MSK Security Group: $SG_MSK"
echo "Redis Security Group: $SG_REDIS"
```

---

## Part 3: Deploy ElastiCache (Redis) (10 minutes)

### Step 6: Create Redis Cluster

```bash
# Create subnet group for ElastiCache
aws elasticache create-cache-subnet-group \
  --cache-subnet-group-name lightning-redis-subnet \
  --cache-subnet-group-description "Lightning Redis subnet group" \
  --subnet-ids $SUBNET_PRIVATE_1 $SUBNET_PRIVATE_2

# Create Redis cluster (single node for testing)
aws elasticache create-cache-cluster \
  --cache-cluster-id lightning-redis \
  --engine redis \
  --cache-node-type cache.t3.micro \
  --num-cache-nodes 1 \
  --cache-subnet-group-name lightning-redis-subnet \
  --security-group-ids $SG_REDIS \
  --engine-version 7.0 \
  --port 6379

# Wait for cluster to be available (~5 minutes)
echo "Waiting for Redis cluster to be available..."
aws elasticache wait cache-cluster-available \
  --cache-cluster-id lightning-redis

# Get Redis endpoint
REDIS_ENDPOINT=$(aws elasticache describe-cache-clusters \
  --cache-cluster-id lightning-redis \
  --show-cache-node-info \
  --query 'CacheClusters[0].CacheNodes[0].Endpoint.Address' \
  --output text)

echo "Redis Endpoint: $REDIS_ENDPOINT"
```

---

## Part 4: Deploy MSK (Kafka) (20 minutes)

### Step 7: Create MSK Cluster

```bash
# Create MSK configuration (optional - custom settings)
cat > msk-config.json << EOF
{
  "ServerProperties": "auto.create.topics.enable=true\ndefault.replication.factor=2\nmin.insync.replicas=1\nnum.partitions=10\nlog.retention.hours=24"
}
EOF

CONFIG_ARN=$(aws kafka create-configuration \
  --name lightning-msk-config \
  --server-properties file://msk-config.json \
  --kafka-versions "3.5.1" \
  --query 'Arn' \
  --output text)

# Create MSK cluster (2 brokers for HA)
cat > msk-cluster.json << EOF
{
  "BrokerNodeGroupInfo": {
    "InstanceType": "kafka.t3.small",
    "ClientSubnets": [
      "$SUBNET_PRIVATE_1",
      "$SUBNET_PRIVATE_2"
    ],
    "SecurityGroups": ["$SG_MSK"],
    "StorageInfo": {
      "EbsStorageInfo": {
        "VolumeSize": 100
      }
    }
  },
  "ClusterName": "lightning-msk",
  "KafkaVersion": "3.5.1",
  "NumberOfBrokerNodes": 2,
  "EncryptionInfo": {
    "EncryptionInTransit": {
      "ClientBroker": "PLAINTEXT",
      "InCluster": false
    }
  },
  "ConfigurationInfo": {
    "Arn": "$CONFIG_ARN",
    "Revision": 1
  }
}
EOF

# Create cluster
CLUSTER_ARN=$(aws kafka create-cluster \
  --cli-input-json file://msk-cluster.json \
  --query 'ClusterArn' \
  --output text)

echo "MSK Cluster ARN: $CLUSTER_ARN"

# Wait for cluster to be active (~15 minutes)
echo "Waiting for MSK cluster to be active (this takes ~15 minutes)..."
aws kafka wait cluster-running --cluster-arn $CLUSTER_ARN

# Get bootstrap servers
KAFKA_BROKERS=$(aws kafka get-bootstrap-brokers \
  --cluster-arn $CLUSTER_ARN \
  --query 'BootstrapBrokerString' \
  --output text)

echo "Kafka Bootstrap Servers: $KAFKA_BROKERS"

# Cleanup temp files
rm msk-config.json msk-cluster.json
```

---

## Part 5: Upload Models to S3 (5 minutes)

### Step 8: Create S3 Bucket and Upload Models

```bash
# Create S3 bucket
BUCKET_NAME="lightning-models-$(date +%s)"
aws s3 mb s3://$BUCKET_NAME --region us-east-1

echo "S3 Bucket: $BUCKET_NAME"

# Upload trained models
aws s3 sync data/models/ s3://$BUCKET_NAME/models/ \
  --exclude "*" \
  --include "stage*.ubj" \
  --include "stage*_metadata*.json" \
  --include "tuned_thresholds*.json"

# Verify upload
aws s3 ls s3://$BUCKET_NAME/models/

# Make bucket private (default, but verify)
aws s3api put-public-access-block \
  --bucket $BUCKET_NAME \
  --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

---

## Part 6: Launch EC2 Instances (15 minutes)

### Step 9: Create EC2 Launch Template

```bash
# Get latest Amazon Linux 2023 AMI
AMI_ID=$(aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=al2023-ami-*-x86_64" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
  --output text)

echo "AMI ID: $AMI_ID"

# Create IAM role for EC2 (to access S3)
cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

aws iam create-role \
  --role-name lightning-ec2-role \
  --assume-role-policy-document file://trust-policy.json

# Attach S3 read policy
aws iam attach-role-policy \
  --role-name lightning-ec2-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# Create instance profile
aws iam create-instance-profile \
  --instance-profile-name lightning-ec2-profile

aws iam add-role-to-instance-profile \
  --instance-profile-name lightning-ec2-profile \
  --role-name lightning-ec2-role

# Wait for profile to propagate
sleep 10

# Create user data script
cat > user-data.sh << 'EOF'
#!/bin/bash
set -e

# Update system
yum update -y

# Install Python 3.11
yum install -y python3.11 python3.11-pip git

# Install system dependencies
yum install -y gcc gcc-c++ make

# Create app directory
mkdir -p /opt/lightning
cd /opt/lightning

# Clone repository (replace with your repo)
# git clone <your-repo-url> .

# For now, we'll download code manually after instance launch

# Install Python dependencies
# python3.11 -m pip install -r requirements.txt
# python3.11 -m pip install -r requirements-streaming.txt

echo "Setup complete!"
EOF

# Launch Consumer EC2 instance
CONSUMER_INSTANCE=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type t3.medium \
  --key-name lightning-key \
  --security-group-ids $SG_EC2 \
  --subnet-id $SUBNET_PUBLIC \
  --iam-instance-profile Name=lightning-ec2-profile \
  --user-data file://user-data.sh \
  --associate-public-ip-address \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=lightning-consumer}]' \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Consumer Instance: $CONSUMER_INSTANCE"

# Launch Producer EC2 instance
PRODUCER_INSTANCE=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type t3.small \
  --key-name lightning-key \
  --security-group-ids $SG_EC2 \
  --subnet-id $SUBNET_PUBLIC \
  --iam-instance-profile Name=lightning-ec2-profile \
  --user-data file://user-data.sh \
  --associate-public-ip-address \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=lightning-producer}]' \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Producer Instance: $PRODUCER_INSTANCE"

# Wait for instances to be running
echo "Waiting for instances to be running..."
aws ec2 wait instance-running --instance-ids $CONSUMER_INSTANCE $PRODUCER_INSTANCE

# Get public IPs
CONSUMER_IP=$(aws ec2 describe-instances \
  --instance-ids $CONSUMER_INSTANCE \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

PRODUCER_IP=$(aws ec2 describe-instances \
  --instance-ids $PRODUCER_INSTANCE \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo ""
echo "=========================================="
echo "EC2 Instances Launched!"
echo "=========================================="
echo "Consumer IP: $CONSUMER_IP"
echo "Producer IP: $PRODUCER_IP"
echo ""
echo "SSH commands:"
echo "  Consumer: ssh -i ~/.ssh/lightning-key.pem ec2-user@$CONSUMER_IP"
echo "  Producer: ssh -i ~/.ssh/lightning-key.pem ec2-user@$PRODUCER_IP"
echo "=========================================="

# Cleanup temp files
rm trust-policy.json user-data.sh
```

---

## Part 7: Deploy Application Code (10 minutes)

### Step 10: Upload Code to EC2

```bash
# Package your code
cd /Users/wittenpan/Desktop/Lightning-Project
tar -czf lightning-app.tar.gz \
  src/ \
  configs/ \
  requirements.txt \
  requirements-streaming.txt

# Copy to Consumer
scp -i ~/.ssh/lightning-key.pem lightning-app.tar.gz ec2-user@$CONSUMER_IP:/home/ec2-user/

# Copy to Producer
scp -i ~/.ssh/lightning-key.pem lightning-app.tar.gz ec2-user@$PRODUCER_IP:/home/ec2-user/

# Cleanup
rm lightning-app.tar.gz
```

### Step 11: Setup Consumer Instance

```bash
# SSH to consumer
ssh -i ~/.ssh/lightning-key.pem ec2-user@$CONSUMER_IP << 'ENDSSH'
# Extract code
cd ~
tar -xzf lightning-app.tar.gz
rm lightning-app.tar.gz

# Install dependencies
python3.11 -m pip install --user -r requirements.txt
python3.11 -m pip install --user -r requirements-streaming.txt

# Download models from S3
aws s3 sync s3://BUCKET_NAME/models/ ~/data/models/

# Create systemd service for consumer
sudo tee /etc/systemd/system/lightning-consumer.service > /dev/null << 'EOF'
[Unit]
Description=Lightning Prediction Consumer
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user
Environment="KAFKA_BROKERS=KAFKA_BROKERS_PLACEHOLDER"
Environment="REDIS_HOST=REDIS_ENDPOINT_PLACEHOLDER"
ExecStart=/usr/bin/python3.11 src/streaming/consumer.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Replace placeholders (do this manually or via sed)
echo "Update /etc/systemd/system/lightning-consumer.service with:"
echo "  KAFKA_BROKERS=$KAFKA_BROKERS"
echo "  REDIS_HOST=$REDIS_ENDPOINT"

ENDSSH
```

### Step 12: Setup Producer Instance

```bash
# SSH to producer
ssh -i ~/.ssh/lightning-key.pem ec2-user@$PRODUCER_IP << 'ENDSSH'
# Extract code
cd ~
tar -xzf lightning-app.tar.gz
rm lightning-app.tar.gz

# Install dependencies
python3.11 -m pip install --user -r requirements.txt
python3.11 -m pip install --user -r requirements-streaming.txt

# Create systemd service for producer
sudo tee /etc/systemd/system/lightning-producer.service > /dev/null << 'EOF'
[Unit]
Description=Lightning Strike Producer
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user
Environment="KAFKA_BROKERS=KAFKA_BROKERS_PLACEHOLDER"
ExecStart=/usr/bin/python3.11 src/streaming/producer.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo "Update /etc/systemd/system/lightning-producer.service with:"
echo "  KAFKA_BROKERS=$KAFKA_BROKERS"

ENDSSH
```

---

## Part 8: Configure & Start Services (5 minutes)

### Step 13: Update Configuration and Start

**On Consumer Instance:**

```bash
ssh -i ~/.ssh/lightning-key.pem ec2-user@$CONSUMER_IP

# Update consumer.py with AWS endpoints
sed -i "s/localhost:9092/$KAFKA_BROKERS/g" src/streaming/consumer.py
sed -i "s/localhost/$REDIS_ENDPOINT/g" src/streaming/consumer.py

# Start service
sudo systemctl daemon-reload
sudo systemctl enable lightning-consumer
sudo systemctl start lightning-consumer

# Check status
sudo systemctl status lightning-consumer

# View logs
sudo journalctl -u lightning-consumer -f
```

**On Producer Instance:**

```bash
ssh -i ~/.ssh/lightning-key.pem ec2-user@$PRODUCER_IP

# Update producer.py with AWS endpoints
sed -i "s/localhost:9092/$KAFKA_BROKERS/g" src/streaming/producer.py

# Start service
sudo systemctl daemon-reload
sudo systemctl enable lightning-producer
sudo systemctl start lightning-producer

# Check status
sudo systemctl status lightning-producer

# View logs
sudo journalctl -u lightning-producer -f
```

---

## Part 9: Monitoring & Verification (5 minutes)

### Step 14: Verify Deployment

```bash
# Check MSK topics
aws kafka list-cluster-operations --cluster-arn $CLUSTER_ARN

# Check ElastiCache
aws elasticache describe-cache-clusters --cache-cluster-id lightning-redis

# Check EC2 instances
aws ec2 describe-instances \
  --instance-ids $CONSUMER_INSTANCE $PRODUCER_INSTANCE \
  --query 'Reservations[].Instances[].[InstanceId,State.Name,PublicIpAddress]' \
  --output table

# View CloudWatch Logs (if configured)
# aws logs tail /aws/ec2/lightning-consumer --follow
```

### Step 15: Test End-to-End

```bash
# SSH to consumer and check logs
ssh -i ~/.ssh/lightning-key.pem ec2-user@$CONSUMER_IP \
  "sudo journalctl -u lightning-consumer -n 50"

# Should see:
# ✓ Loaded models from data/models
# ✓ Connected to Redis: <redis-endpoint>
# Consumer initialized: lightning-strikes → lightning-predictions
# Strikes processed: 100
```

---

## Part 10: Cost Optimization

### Reduce Costs for Testing

```bash
# Use spot instances for EC2 (50-90% cheaper)
aws ec2 request-spot-instances \
  --spot-price "0.05" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification file://spot-spec.json

# Use smaller instance types
# - EC2: t3.micro instead of t3.medium
# - MSK: kafka.m5.large (can't go smaller)
# - Redis: cache.t3.micro (smallest)

# Set auto-shutdown for EC2
# Add to user-data.sh:
echo "sudo shutdown -h +60" >> /etc/rc.local  # Shutdown after 1 hour
```

### Estimated Monthly Costs

| Service | Instance Type | Cost/Month |
|---------|---------------|------------|
| MSK (2 brokers) | kafka.t3.small | ~$100 |
| ElastiCache | cache.t3.micro | ~$12 |
| EC2 Consumer | t3.medium | ~$30 |
| EC2 Producer | t3.small | ~$15 |
| S3 Storage | <1 GB | ~$1 |
| Data Transfer | <10 GB | ~$1 |
| **Total** | | **~$159/month** |

**With spot instances: ~$80-100/month**

---

## Cleanup (Delete Everything)

### **⚠️ WARNING: This will delete all resources!**

```bash
# Stop EC2 instances
aws ec2 terminate-instances \
  --instance-ids $CONSUMER_INSTANCE $PRODUCER_INSTANCE

# Delete MSK cluster
aws kafka delete-cluster --cluster-arn $CLUSTER_ARN

# Delete ElastiCache cluster
aws elasticache delete-cache-cluster \
  --cache-cluster-id lightning-redis

# Delete S3 bucket
aws s3 rb s3://$BUCKET_NAME --force

# Delete security groups (after instances/clusters are deleted)
sleep 300  # Wait 5 minutes
aws ec2 delete-security-group --group-id $SG_EC2
aws ec2 delete-security-group --group-id $SG_MSK
aws ec2 delete-security-group --group-id $SG_REDIS

# Delete subnets
aws ec2 delete-subnet --subnet-id $SUBNET_PUBLIC
aws ec2 delete-subnet --subnet-id $SUBNET_PRIVATE_1
aws ec2 delete-subnet --subnet-id $SUBNET_PRIVATE_2

# Detach and delete Internet Gateway
aws ec2 detach-internet-gateway --internet-gateway-id $IGW_ID --vpc-id $VPC_ID
aws ec2 delete-internet-gateway --internet-gateway-id $IGW_ID

# Delete route table
aws ec2 delete-route-table --route-table-id $RTB_ID

# Delete VPC
aws ec2 delete-vpc --vpc-id $VPC_ID

# Delete IAM resources
aws iam remove-role-from-instance-profile \
  --instance-profile-name lightning-ec2-profile \
  --role-name lightning-ec2-role
aws iam delete-instance-profile --instance-profile-name lightning-ec2-profile
aws iam detach-role-policy \
  --role-name lightning-ec2-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
aws iam delete-role --role-name lightning-ec2-role

echo "✓ All resources deleted!"
```

---

## Troubleshooting

### Can't connect to MSK from EC2

```bash
# Verify security group allows EC2 → MSK
aws ec2 describe-security-groups --group-ids $SG_MSK

# Test connection from EC2
ssh -i ~/.ssh/lightning-key.pem ec2-user@$CONSUMER_IP
telnet <kafka-broker-endpoint> 9092
```

### Can't connect to Redis from EC2

```bash
# Verify security group
aws ec2 describe-security-groups --group-ids $SG_REDIS

# Test from EC2
ssh -i ~/.ssh/lightning-key.pem ec2-user@$CONSUMER_IP
redis-cli -h $REDIS_ENDPOINT ping
```

### Models not loading

```bash
# Check S3 permissions
aws s3 ls s3://$BUCKET_NAME/models/

# Check EC2 IAM role
aws sts get-caller-identity  # Run on EC2

# Manually download
aws s3 sync s3://$BUCKET_NAME/models/ ~/data/models/
```

---

## Alternative: Quick Deploy with Docker on Single EC2

For simpler testing, run everything on one EC2 instance:

```bash
# Launch larger instance
aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type t3.xlarge \
  --key-name lightning-key \
  --security-group-ids $SG_EC2 \
  --subnet-id $SUBNET_PUBLIC \
  --associate-public-ip-address \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=lightning-all-in-one}]'

# SSH and install Docker
ssh -i ~/.ssh/lightning-key.pem ec2-user@<INSTANCE-IP>

# Install Docker
sudo yum install -y docker
sudo systemctl start docker
sudo usermod -aG docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Upload and run your docker-compose.yml
scp -i ~/.ssh/lightning-key.pem docker-compose.yml ec2-user@<INSTANCE-IP>:~/
ssh -i ~/.ssh/lightning-key.pem ec2-user@<INSTANCE-IP>
docker-compose up -d
```

**Cost: ~$70/month for t3.xlarge**

---

## Next Steps

1. **Set up CloudWatch monitoring** for metrics
2. **Configure auto-scaling** for EC2 instances
3. **Add Application Load Balancer** for multiple consumers
4. **Implement CI/CD** with CodePipeline
5. **Add SNS alerts** for failures

Good luck with your AWS deployment! 🚀
