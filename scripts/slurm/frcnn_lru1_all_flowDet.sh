#!/bin/bash

# Base project root (2 levels up from slurm script)
ROOT="/home/chen"
PROJECT_ROOT="${ROOT}/TMNF/examples/flowDet"

# Run extract script
#bash "$PROJECT_ROOT/extract_feat_custom.sh" "lru1" $ROOT
#bash "$PROJECT_ROOT/extract_feat_custom.sh" "lru1_drone" $ROOT
#bash "$PROJECT_ROOT/extract_feat_custom.sh" "lru1_lander" $ROOT
#bash "$PROJECT_ROOT/extract_feat_custom.sh" "lru1_lru2" $ROOT

# Run train and test flowDet script
bash "$PROJECT_ROOT/train_onlyFlow_custom.sh" "lru1"
bash "$PROJECT_ROOT/train_onlyFlow_custom.sh" "lru1_drone"
bash "$PROJECT_ROOT/train_onlyFlow_custom.sh" "lru1_lander"
bash "$PROJECT_ROOT/train_onlyFlow_custom.sh" "lru1_lru2"