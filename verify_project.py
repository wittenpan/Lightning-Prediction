#!/usr/bin/env python3
"""
Project Verification Script

Checks that all files, dependencies, and services are properly configured.
Run this before deployment to catch any issues.
"""

import sys
import os
from pathlib import Path
import subprocess

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

def check(name, passed):
    """Print check result."""
    if passed:
        print(f"{GREEN}✓{RESET} {name}")
        return True
    else:
        print(f"{RED}✗{RESET} {name}")
        return False

def warn(message):
    """Print warning."""
    print(f"{YELLOW}⚠{RESET}  {message}")

def header(text):
    """Print section header."""
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}{text}{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")

def main():
    print(f"\n{BOLD}Lightning Prediction System - Project Verification{RESET}\n")

    all_passed = True
    warnings = []

    # ====================
    # 1. DIRECTORY STRUCTURE
    # ====================
    header("1. Directory Structure")

    required_dirs = [
        'src/ml',
        'src/processing',
        'src/streaming',
        'data/models',
        'data/processed',
        'data/raw',
        'configs',
        'notebooks'
    ]

    for dir_path in required_dirs:
        exists = Path(dir_path).exists()
        all_passed &= check(f"Directory: {dir_path}", exists)
        if not exists:
            warnings.append(f"Create missing directory: mkdir -p {dir_path}")

    # ====================
    # 2. PYTHON FILES
    # ====================
    header("2. Python Source Files")

    required_files = [
        # ML
        'src/ml/__init__.py',
        'src/ml/train_stage1.py',
        'src/ml/train_stage2.py',
        'src/ml/eval.py',
        'src/ml/eval_two_stage.py',
        'src/ml/tune_threshold.py',

        # Processing
        'src/processing/__init__.py',
        'src/processing/config_loader.py',
        'src/processing/grid_manager.py',
        'src/processing/feature_engineering.py',
        'src/processing/data_preparation.py',

        # Streaming
        'src/streaming/__init__.py',
        'src/streaming/producer.py',
        'src/streaming/consumer.py',
        'src/streaming/redis_cache.py',
    ]

    for file_path in required_files:
        exists = Path(file_path).exists()
        all_passed &= check(f"File: {file_path}", exists)

    # ====================
    # 3. CONFIGURATION FILES
    # ====================
    header("3. Configuration Files")

    config_files = [
        'configs/grid_config.yml',
        'docker-compose.yml',
        'requirements.txt',
        'requirements-streaming.txt',
    ]

    for file_path in config_files:
        exists = Path(file_path).exists()
        all_passed &= check(f"Config: {file_path}", exists)

    # ====================
    # 4. DOCUMENTATION
    # ====================
    header("4. Documentation")

    docs = [
        'DEPLOYMENT_GUIDE.md',
        'STREAMING_README.md',
        'PROJECT_SUMMARY.md',
    ]

    for doc in docs:
        exists = Path(doc).exists()
        check(f"Doc: {doc}", exists)
        if not exists:
            warnings.append(f"Missing documentation: {doc}")

    # ====================
    # 5. TRAINED MODELS
    # ====================
    header("5. Trained Models")

    models = [
        'data/models/stage1_model_15min.ubj',
        'data/models/stage2_model_15min.ubj',
        'data/models/stage1_metadata_15min.json',
        'data/models/stage2_metadata_15min.json',
    ]

    models_exist = True
    for model in models:
        exists = Path(model).exists()
        models_exist &= exists
        check(f"Model: {model}", exists)

    if not models_exist:
        warnings.append("Train models: python -m src.ml.train_stage1 --horizon 15min")
        warnings.append("Train models: python -m src.ml.train_stage2 --horizon 15min")

    # ====================
    # 6. PYTHON DEPENDENCIES
    # ====================
    header("6. Python Dependencies (ML)")

    ml_deps = [
        'pandas',
        'numpy',
        'xgboost',
        'scikit-learn',
        'h3',
        'pyyaml'
    ]

    for dep in ml_deps:
        try:
            __import__(dep)
            check(f"Package: {dep}", True)
        except ImportError:
            all_passed = False
            check(f"Package: {dep}", False)
            warnings.append(f"Install ML dependencies: pip install -r requirements.txt")
            break  # Only warn once

    # ====================
    # 7. STREAMING DEPENDENCIES
    # ====================
    header("7. Python Dependencies (Streaming)")

    streaming_deps = [
        ('kafka', 'kafka-python'),
        ('redis', 'redis'),
        ('websockets', 'websockets'),
    ]

    streaming_ok = True
    for module_name, package_name in streaming_deps:
        try:
            __import__(module_name)
            check(f"Package: {package_name}", True)
        except ImportError:
            streaming_ok = False
            check(f"Package: {package_name}", False)

    if not streaming_ok:
        warnings.append("Install streaming dependencies: pip install -r requirements-streaming.txt")

    # ====================
    # 8. DOCKER SERVICES
    # ====================
    header("8. Docker Services")

    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True, timeout=5)
        docker_running = result.returncode == 0
        check("Docker daemon running", docker_running)

        if docker_running:
            # Check if containers are running
            output = result.stdout
            kafka_running = 'lightning-kafka' in output
            redis_running = 'lightning-redis' in output

            check("Kafka container running", kafka_running)
            check("Redis container running", redis_running)

            if not (kafka_running and redis_running):
                warnings.append("Start Docker services: docker-compose up -d")
        else:
            warnings.append("Start Docker Desktop and run: docker-compose up -d")

    except (subprocess.TimeoutExpired, FileNotFoundError):
        check("Docker available", False)
        warnings.append("Install Docker Desktop from https://www.docker.com/products/docker-desktop")

    # ====================
    # 9. PROCESSED DATA
    # ====================
    header("9. Processed Data")

    processed_files = list(Path('data/processed').glob('*.parquet'))
    has_data = len(processed_files) > 0

    check(f"Processed data files ({len(processed_files)} files)", has_data)

    if not has_data:
        warnings.append("Process data: python -m src.processing.data_preparation")

    # ====================
    # 10. GIT REPOSITORY
    # ====================
    header("10. Git Repository")

    git_exists = Path('.git').exists()
    check("Git repository initialized", git_exists)

    if git_exists:
        try:
            result = subprocess.run(['git', 'status'], capture_output=True, text=True)
            if result.returncode == 0:
                check("Git status clean", 'nothing to commit' in result.stdout)
        except:
            pass

    # ====================
    # SUMMARY
    # ====================
    header("Verification Summary")

    if all_passed and not warnings:
        print(f"\n{GREEN}{BOLD}✅ ALL CHECKS PASSED!{RESET}")
        print(f"\n{BOLD}Your project is ready for deployment.{RESET}")
        print(f"\nNext steps:")
        print(f"  1. docker-compose up -d")
        print(f"  2. python src/streaming/consumer.py")
        print(f"  3. python src/streaming/producer.py")

    else:
        print(f"\n{YELLOW}{BOLD}⚠  SOME CHECKS FAILED{RESET}")

        if warnings:
            print(f"\n{BOLD}Recommended fixes:{RESET}")
            for i, warning in enumerate(set(warnings), 1):
                print(f"  {i}. {warning}")

        print(f"\n{BOLD}See DEPLOYMENT_GUIDE.md for detailed instructions.{RESET}")

    return 0 if (all_passed and not warnings) else 1


if __name__ == "__main__":
    sys.exit(main())
