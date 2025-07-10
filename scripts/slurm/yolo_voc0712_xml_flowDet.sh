#!/bin/bash

# Base project root (2 levels up from slurm script)
#PROJECT_ROOT="/home/chen_le/TMNF/examples/flowDet"
PROJECT_ROOT="/home/chen/TMNF"

# Run extract script
#bash "$PROJECT_ROOT/scripts/YOLOv8/extraction/extract_voc0712.sh"

# Run train and test flowDet script
bash "$PROJECT_ROOT/examples/flowDet/train_onlyFlow_yolo.sh" "xml"
