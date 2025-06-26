#!/bin/bash

# Base project root (2 levels up from slurm script)
#PROJECT_ROOT="/home/chen_le/TMNF/examples/flowDet"
PROJECT_ROOT="/home/chen/TMNF/examples/flowDet"

# Run extract script
#bash "$PROJECT_ROOT/extract_feat_custom.sh" "xml"

# Run train and test flowDet script
bash "$PROJECT_ROOT/train_onlyFlow_custom.sh" "xml"
