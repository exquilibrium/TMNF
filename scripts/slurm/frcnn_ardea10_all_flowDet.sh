#!/bin/bash

# Base project root (2 levels up from slurm script)
#ROOT="/home/chen"
ROOT="/home/chen_le"
PROJECT_ROOT="${ROOT}/TMNF/examples/flowDet"

# Run extract script
bash "$PROJECT_ROOT/extract_feat_custom.sh" "ardea10" $ROOT
bash "$PROJECT_ROOT/extract_feat_custom.sh" "ardea10_lander" $ROOT
bash "$PROJECT_ROOT/extract_feat_custom.sh" "ardea10_lru1" $ROOT
bash "$PROJECT_ROOT/extract_feat_custom.sh" "ardea10_lru2" $ROOT

# Run train and test flowDet script
bash "$PROJECT_ROOT/train_onlyFlow_custom.sh" "ardea10"
bash "$PROJECT_ROOT/train_onlyFlow_custom.sh" "ardea10_lander"
bash "$PROJECT_ROOT/train_onlyFlow_custom.sh" "ardea10_lru1"
bash "$PROJECT_ROOT/train_onlyFlow_custom.sh" "ardea10_lru2"