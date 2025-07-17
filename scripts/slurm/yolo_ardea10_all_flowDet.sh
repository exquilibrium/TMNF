#!/bin/bash

# Base project root (2 levels up from slurm script)
#PROJECT_ROOT="/home/chen/TMNF"
PROJECT_ROOT="/home/chen_le/TMNF"

# Run extract script
bash "$PROJECT_ROOT/scripts/YOLOv8/extraction/extract_ardea10.sh"
bash "$PROJECT_ROOT/scripts/YOLOv8/extraction/extract_ardea10_lander.sh"
bash "$PROJECT_ROOT/scripts/YOLOv8/extraction/extract_ardea10_lru1.sh"
bash "$PROJECT_ROOT/scripts/YOLOv8/extraction/extract_ardea10_lru2.sh"

# Run train and test flowDet script
bash "$PROJECT_ROOT/examples/flowDet/train_onlyFlow_yolo.sh" "ardea10"
bash "$PROJECT_ROOT/examples/flowDet/train_onlyFlow_yolo.sh" "ardea10_lander"
bash "$PROJECT_ROOT/examples/flowDet/train_onlyFlow_yolo.sh" "ardea10_lru1"
bash "$PROJECT_ROOT/examples/flowDet/train_onlyFlow_yolo.sh" "ardea10_lru2"
