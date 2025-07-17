#!/bin/bash

# Base project root (2 levels up from slurm script)
#ROOT="/home/chen"
ROOT="/home/chen_le"
PROJECT_ROOT="${ROOT}/TMNF/examples/flowDet"

# Run extract script
bash "$PROJECT_ROOT/extract_feat_custom.sh" "xml" $ROOT

# Run train and test flowDet script
bash "$PROJECT_ROOT/train_onlyFlow_custom.sh" "xml"
