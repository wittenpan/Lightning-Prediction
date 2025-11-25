#!/bin/bash
#
# AWS Deployment Script for Lightning Prediction System
#
# This script automates the deployment of:
# - VPC and networking
# - ElastiCache (Redis)
# - MSK (Kafka)
# - EC2 instances (Consumer + Producer)
# - S3 bucket (for models)
#
# Usage: ./deploy_aws.sh
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REGION="us-east-1"
PROJECT_NAME="lightning"
KEY_NAME="${PROJECT_NAME}-key"

echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Lightning Prediction System - AWS Deployment      ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command -v aws &> /dev/null; then
    echo -e "${RED}✗ AWS CLI not installed${NC}"
    echo "Install with: brew install awscli"
    exit 1
fi
echo -e "${GREEN}✓ AWS CLI installed${NC}"

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}✗ AWS credentials not configured${NC}"
    echo "Run: aws configure"
    exit 1
fi
echo -e "${GREEN}✓ AWS credentials configured${NC}"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo -e "${GREEN}✓ AWS Account: ${ACCOUNT_ID}${NC}"

# Save configuration
CONFIG_FILE=".aws-deployment-config"

echo ""
echo -e "${YELLOW}Step 1: Creating VPC and Networking...${NC}"

# Create VPC
VPC_ID=$(aws ec2 create-vpc \
  --cidr-block 10.0.0.0/16 \
  --tag-specifications "ResourceType=vpc,Tags=[{Key=Name,Value=${PROJECT_NAME}-vpc}]" \
  --query 'Vpc.VpcId' \
  --output text \
  --region $REGION)

echo "VPC_ID=$VPC_ID" >> $CONFIG_FILE
echo -e "${GREEN}✓ VPC created: ${VPC_ID}${NC}"

# Enable DNS
aws ec2 modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-hostnames --region $REGION
aws ec2 modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-support --region $REGION

# Create Internet Gateway
IGW_ID=$(aws ec2 create-internet-gateway \
  --tag-specifications "ResourceType=internet-gateway,Tags=[{Key=Name,Value=${PROJECT_NAME}-igw}]" \
  --query 'InternetGateway.InternetGatewayId' \
  --output text \
  --region $REGION)

echo "IGW_ID=$IGW_ID" >> $CONFIG_FILE
aws ec2 attach-internet-gateway --vpc-id $VPC_ID --internet-gateway-id $IGW_ID --region $REGION
echo -e "${GREEN}✓ Internet Gateway created and attached${NC}"

# Create Subnets
SUBNET_PUBLIC=$(aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block 10.0.1.0/24 \
  --availability-zone ${REGION}a \
  --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=${PROJECT_NAME}-public}]" \
  --query 'Subnet.SubnetId' \
  --output text \
  --region $REGION)

SUBNET_PRIVATE_1=$(aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block 10.0.2.0/24 \
  --availability-zone ${REGION}a \
  --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=${PROJECT_NAME}-private-1}]" \
  --query 'Subnet.SubnetId' \
  --output text \
  --region $REGION)

SUBNET_PRIVATE_2=$(aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block 10.0.3.0/24 \
  --availability-zone ${REGION}b \
  --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=${PROJECT_NAME}-private-2}]" \
  --query 'Subnet.SubnetId' \
  --output text \
  --region $REGION)

echo "SUBNET_PUBLIC=$SUBNET_PUBLIC" >> $CONFIG_FILE
echo "SUBNET_PRIVATE_1=$SUBNET_PRIVATE_1" >> $CONFIG_FILE
echo "SUBNET_PRIVATE_2=$SUBNET_PRIVATE_2" >> $CONFIG_FILE
echo -e "${GREEN}✓ Subnets created${NC}"

# Create and configure route table
RTB_ID=$(aws ec2 create-route-table \
  --vpc-id $VPC_ID \
  --tag-specifications "ResourceType=route-table,Tags=[{Key=Name,Value=${PROJECT_NAME}-rtb}]" \
  --query 'RouteTable.RouteTableId' \
  --output text \
  --region $REGION)

aws ec2 create-route --route-table-id $RTB_ID --destination-cidr-block 0.0.0.0/0 --gateway-id $IGW_ID --region $REGION
aws ec2 associate-route-table --subnet-id $SUBNET_PUBLIC --route-table-id $RTB_ID --region $REGION &> /dev/null
echo -e "${GREEN}✓ Route table configured${NC}"

echo ""
echo -e "${YELLOW}Step 2: Creating Security Groups...${NC}"

# Create security groups
SG_EC2=$(aws ec2 create-security-group \
  --group-name ${PROJECT_NAME}-ec2-sg \
  --description "Lightning EC2 instances" \
  --vpc-id $VPC_ID \
  --query 'GroupId' \
  --output text \
  --region $REGION)

aws ec2 authorize-security-group-ingress --group-id $SG_EC2 --protocol tcp --port 22 --cidr 0.0.0.0/0 --region $REGION
echo "SG_EC2=$SG_EC2" >> $CONFIG_FILE
echo -e "${GREEN}✓ EC2 security group created${NC}"

SG_REDIS=$(aws ec2 create-security-group \
  --group-name ${PROJECT_NAME}-redis-sg \
  --description "Lightning Redis cluster" \
  --vpc-id $VPC_ID \
  --query 'GroupId' \
  --output text \
  --region $REGION)

aws ec2 authorize-security-group-ingress --group-id $SG_REDIS --protocol tcp --port 6379 --source-group $SG_EC2 --region $REGION
echo "SG_REDIS=$SG_REDIS" >> $CONFIG_FILE
echo -e "${GREEN}✓ Redis security group created${NC}"

echo ""
echo -e "${YELLOW}Step 3: Creating SSH Key Pair...${NC}"

# Create key pair
if [ ! -f ~/.ssh/${KEY_NAME}.pem ]; then
    aws ec2 create-key-pair \
      --key-name $KEY_NAME \
      --query 'KeyMaterial' \
      --output text \
      --region $REGION > ~/.ssh/${KEY_NAME}.pem

    chmod 400 ~/.ssh/${KEY_NAME}.pem
    echo -e "${GREEN}✓ SSH key created: ~/.ssh/${KEY_NAME}.pem${NC}"
else
    echo -e "${YELLOW}⚠ SSH key already exists${NC}"
fi

echo ""
echo -e "${YELLOW}Step 4: Creating Redis Cluster (ElastiCache)...${NC}"

# Create Redis subnet group
aws elasticache create-cache-subnet-group \
  --cache-subnet-group-name ${PROJECT_NAME}-redis-subnet \
  --cache-subnet-group-description "Lightning Redis subnet group" \
  --subnet-ids $SUBNET_PRIVATE_1 $SUBNET_PRIVATE_2 \
  --region $REGION &> /dev/null || echo "Subnet group may already exist"

# Create Redis cluster
aws elasticache create-cache-cluster \
  --cache-cluster-id ${PROJECT_NAME}-redis \
  --engine redis \
  --cache-node-type cache.t3.micro \
  --num-cache-nodes 1 \
  --cache-subnet-group-name ${PROJECT_NAME}-redis-subnet \
  --security-group-ids $SG_REDIS \
  --engine-version 7.0 \
  --port 6379 \
  --region $REGION &> /dev/null || echo "Redis cluster may already exist"

echo -e "${YELLOW}⏳ Waiting for Redis to be available (this takes ~5 minutes)...${NC}"
aws elasticache wait cache-cluster-available --cache-cluster-id ${PROJECT_NAME}-redis --region $REGION

REDIS_ENDPOINT=$(aws elasticache describe-cache-clusters \
  --cache-cluster-id ${PROJECT_NAME}-redis \
  --show-cache-node-info \
  --query 'CacheClusters[0].CacheNodes[0].Endpoint.Address' \
  --output text \
  --region $REGION)

echo "REDIS_ENDPOINT=$REDIS_ENDPOINT" >> $CONFIG_FILE
echo -e "${GREEN}✓ Redis cluster ready: ${REDIS_ENDPOINT}${NC}"

echo ""
echo -e "${YELLOW}Step 5: Creating S3 Bucket for Models...${NC}"

BUCKET_NAME="${PROJECT_NAME}-models-${ACCOUNT_ID}"
aws s3 mb s3://$BUCKET_NAME --region $REGION 2>/dev/null || echo "Bucket may already exist"

# Upload models
if [ -d "data/models" ]; then
    aws s3 sync data/models/ s3://$BUCKET_NAME/models/ \
      --exclude "*" \
      --include "stage*.ubj" \
      --include "*_metadata*.json" \
      --include "tuned_thresholds*.json" \
      --region $REGION
    echo -e "${GREEN}✓ Models uploaded to S3${NC}"
else
    echo -e "${YELLOW}⚠ data/models directory not found - models not uploaded${NC}"
fi

echo "BUCKET_NAME=$BUCKET_NAME" >> $CONFIG_FILE

echo ""
echo -e "${YELLOW}Step 6: Creating IAM Role for EC2...${NC}"

# Create IAM role
cat > /tmp/trust-policy.json << EOF
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
  --role-name ${PROJECT_NAME}-ec2-role \
  --assume-role-policy-document file:///tmp/trust-policy.json \
  --region $REGION &> /dev/null || echo "Role may already exist"

aws iam attach-role-policy \
  --role-name ${PROJECT_NAME}-ec2-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess \
  --region $REGION &> /dev/null || true

aws iam create-instance-profile \
  --instance-profile-name ${PROJECT_NAME}-ec2-profile \
  --region $REGION &> /dev/null || echo "Profile may already exist"

aws iam add-role-to-instance-profile \
  --instance-profile-name ${PROJECT_NAME}-ec2-profile \
  --role-name ${PROJECT_NAME}-ec2-role \
  --region $REGION &> /dev/null || true

sleep 10  # Wait for IAM propagation
echo -e "${GREEN}✓ IAM role created${NC}"

rm /tmp/trust-policy.json

echo ""
echo -e "${YELLOW}Step 7: Launching EC2 Instance (All-in-One)...${NC}"

# Get Amazon Linux 2023 AMI
AMI_ID=$(aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=al2023-ami-*-x86_64" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
  --output text \
  --region $REGION)

# Create user data script
cat > /tmp/user-data.sh << 'EOF'
#!/bin/bash
set -e

# Update system
yum update -y

# Install Docker
yum install -y docker git
systemctl start docker
systemctl enable docker
usermod -aG docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Create app directory
mkdir -p /opt/lightning
chown ec2-user:ec2-user /opt/lightning

echo "Setup complete!" > /tmp/setup-complete.txt
EOF

# Launch instance
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type t3.large \
  --key-name $KEY_NAME \
  --security-group-ids $SG_EC2 \
  --subnet-id $SUBNET_PUBLIC \
  --iam-instance-profile Name=${PROJECT_NAME}-ec2-profile \
  --user-data file:///tmp/user-data.sh \
  --associate-public-ip-address \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${PROJECT_NAME}-all-in-one}]" \
  --query 'Instances[0].InstanceId' \
  --output text \
  --region $REGION)

echo "INSTANCE_ID=$INSTANCE_ID" >> $CONFIG_FILE
echo -e "${GREEN}✓ EC2 instance launching: ${INSTANCE_ID}${NC}"

echo -e "${YELLOW}⏳ Waiting for instance to be running...${NC}"
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

# Get public IP
INSTANCE_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text \
  --region $REGION)

echo "INSTANCE_IP=$INSTANCE_IP" >> $CONFIG_FILE
echo -e "${GREEN}✓ Instance running: ${INSTANCE_IP}${NC}"

rm /tmp/user-data.sh

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Deployment Complete!                    ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo ""
echo "1. Wait 2 minutes for instance initialization to complete"
echo ""
echo "2. SSH to your instance:"
echo -e "   ${GREEN}ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${INSTANCE_IP}${NC}"
echo ""
echo "3. Upload your code:"
echo -e "   ${GREEN}scp -i ~/.ssh/${KEY_NAME}.pem docker-compose.yml ec2-user@${INSTANCE_IP}:~/lightning/${NC}"
echo -e "   ${GREEN}scp -r -i ~/.ssh/${KEY_NAME}.pem src ec2-user@${INSTANCE_IP}:~/lightning/${NC}"
echo ""
echo "4. On the instance, run Docker Compose:"
echo "   cd ~/lightning"
echo "   docker-compose up -d"
echo ""
echo -e "${YELLOW}Deployment Configuration:${NC}"
echo "  Redis Endpoint: $REDIS_ENDPOINT"
echo "  S3 Bucket: $BUCKET_NAME"
echo "  Instance ID: $INSTANCE_ID"
echo "  Instance IP: $INSTANCE_IP"
echo ""
echo "Configuration saved to: $CONFIG_FILE"
echo ""
echo -e "${YELLOW}To destroy everything:${NC}"
echo "  ./cleanup_aws.sh"
echo ""
