#!/bin/bash

# Base project root (2 levels up from slurm script)
#PROJECT_ROOT="/home/chen_le/TMNF/examples/flowDet"
PROJECT_ROOT="/home/chen/TMNF/examples/flowDet"

# Run extract script
#bash "$PROJECT_ROOT/extract_feat_custom.sh" "lru1"
#bash "$PROJECT_ROOT/extract_feat_custom.sh" "lru1_drone"
#bash "$PROJECT_ROOT/extract_feat_custom.sh" "lru1_lander"
#bash "$PROJECT_ROOT/extract_feat_custom.sh" "lru1_lru2"

# Run train and test flowDet script
bash "$PROJECT_ROOT/train_onlyFlow_custom.sh" "lru1"
bash "$PROJECT_ROOT/train_onlyFlow_custom.sh" "lru1_drone"
bash "$PROJECT_ROOT/train_onlyFlow_custom.sh" "lru1_lander"
bash "$PROJECT_ROOT/train_onlyFlow_custom.sh" "lru1_lru2"