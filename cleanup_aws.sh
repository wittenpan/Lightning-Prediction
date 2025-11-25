#!/bin/bash
#
# AWS Cleanup Script for Lightning Prediction System
#
# This script destroys all AWS resources created by deploy_aws.sh
#
# Usage: ./cleanup_aws.sh
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CONFIG_FILE=".aws-deployment-config"
REGION="us-east-1"

echo -e "${RED}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${RED}║   Lightning Prediction System - AWS Cleanup         ║${NC}"
echo -e "${RED}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# Load configuration
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}✗ Configuration file not found: $CONFIG_FILE${NC}"
    echo "Nothing to clean up."
    exit 1
fi

source $CONFIG_FILE

echo -e "${YELLOW}⚠️  WARNING: This will DELETE all AWS resources!${NC}"
echo ""
echo "Resources to be deleted:"
echo "  - EC2 Instance: $INSTANCE_ID"
echo "  - Redis Cluster: lightning-redis"
echo "  - S3 Bucket: $BUCKET_NAME"
echo "  - VPC and all networking: $VPC_ID"
echo ""
read -p "Are you sure? (type 'yes' to confirm): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo -e "${YELLOW}Starting cleanup...${NC}"

# Terminate EC2 instances
if [ ! -z "$INSTANCE_ID" ]; then
    echo -e "${YELLOW}Terminating EC2 instance...${NC}"
    aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION &> /dev/null || true
    echo -e "${GREEN}✓ EC2 instance terminated${NC}"
fi

# Delete Redis cluster
echo -e "${YELLOW}Deleting Redis cluster...${NC}"
aws elasticache delete-cache-cluster \
  --cache-cluster-id lightning-redis \
  --region $REGION &> /dev/null || true
echo -e "${GREEN}✓ Redis cluster deletion initiated${NC}"

# Delete S3 bucket
if [ ! -z "$BUCKET_NAME" ]; then
    echo -e "${YELLOW}Deleting S3 bucket...${NC}"
    aws s3 rb s3://$BUCKET_NAME --force --region $REGION &> /dev/null || true
    echo -e "${GREEN}✓ S3 bucket deleted${NC}"
fi

# Wait for instances to terminate
if [ ! -z "$INSTANCE_ID" ]; then
    echo -e "${YELLOW}⏳ Waiting for EC2 instances to terminate...${NC}"
    aws ec2 wait instance-terminated --instance-ids $INSTANCE_ID --region $REGION 2>/dev/null || sleep 30
    echo -e "${GREEN}✓ EC2 instances terminated${NC}"
fi

# Delete security groups
echo -e "${YELLOW}Deleting security groups...${NC}"
sleep 30  # Wait for dependencies to clear
aws ec2 delete-security-group --group-id $SG_EC2 --region $REGION &> /dev/null || true
aws ec2 delete-security-group --group-id $SG_REDIS --region $REGION &> /dev/null || true
echo -e "${GREEN}✓ Security groups deleted${NC}"

# Delete ElastiCache subnet group
echo -e "${YELLOW}Deleting ElastiCache subnet group...${NC}"
aws elasticache delete-cache-subnet-group \
  --cache-subnet-group-name lightning-redis-subnet \
  --region $REGION &> /dev/null || true
echo -e "${GREEN}✓ ElastiCache subnet group deleted${NC}"

# Delete subnets
echo -e "${YELLOW}Deleting subnets...${NC}"
aws ec2 delete-subnet --subnet-id $SUBNET_PUBLIC --region $REGION &> /dev/null || true
aws ec2 delete-subnet --subnet-id $SUBNET_PRIVATE_1 --region $REGION &> /dev/null || true
aws ec2 delete-subnet --subnet-id $SUBNET_PRIVATE_2 --region $REGION &> /dev/null || true
echo -e "${GREEN}✓ Subnets deleted${NC}"

# Detach and delete Internet Gateway
if [ ! -z "$IGW_ID" ] && [ ! -z "$VPC_ID" ]; then
    echo -e "${YELLOW}Deleting Internet Gateway...${NC}"
    aws ec2 detach-internet-gateway --internet-gateway-id $IGW_ID --vpc-id $VPC_ID --region $REGION &> /dev/null || true
    aws ec2 delete-internet-gateway --internet-gateway-id $IGW_ID --region $REGION &> /dev/null || true
    echo -e "${GREEN}✓ Internet Gateway deleted${NC}"
fi

# Delete VPC
if [ ! -z "$VPC_ID" ]; then
    echo -e "${YELLOW}Deleting VPC...${NC}"
    sleep 10
    aws ec2 delete-vpc --vpc-id $VPC_ID --region $REGION &> /dev/null || true
    echo -e "${GREEN}✓ VPC deleted${NC}"
fi

# Delete IAM resources
echo -e "${YELLOW}Deleting IAM resources...${NC}"
aws iam remove-role-from-instance-profile \
  --instance-profile-name lightning-ec2-profile \
  --role-name lightning-ec2-role \
  --region $REGION &> /dev/null || true

aws iam delete-instance-profile \
  --instance-profile-name lightning-ec2-profile \
  --region $REGION &> /dev/null || true

aws iam detach-role-policy \
  --role-name lightning-ec2-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess \
  --region $REGION &> /dev/null || true

aws iam delete-role \
  --role-name lightning-ec2-role \
  --region $REGION &> /dev/null || true

echo -e "${GREEN}✓ IAM resources deleted${NC}"

# Delete configuration file
rm $CONFIG_FILE
echo -e "${GREEN}✓ Configuration file deleted${NC}"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Cleanup Complete!                       ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo "All AWS resources have been deleted."
echo ""
echo -e "${YELLOW}Note: Some resources (like ElastiCache) may take a few minutes to fully delete.${NC}"
echo "Check AWS Console to verify: https://console.aws.amazon.com"
echo ""
