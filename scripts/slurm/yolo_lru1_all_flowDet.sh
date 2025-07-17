#!/bin/bash

# Base project root (2 levels up from slurm script)
#PROJECT_ROOT="/home/chen/TMNF"
PROJECT_ROOT="/home/chen_le/TMNF"

# Run extract script
bash "$PROJECT_ROOT/scripts/YOLOv8/extraction/extract_lru1.sh"
bash "$PROJECT_ROOT/scripts/YOLOv8/extraction/extract_lru1_drone.sh"
bash "$PROJECT_ROOT/scripts/YOLOv8/extraction/extract_lru1_lander.sh"
bash "$PROJECT_ROOT/scripts/YOLOv8/extraction/extract_lru1_lru2.sh"

# Run train and test flowDet script
bash "$PROJECT_ROOT/examples/flowDet/train_onlyFlow_yolo.sh" "lru1"
bash "$PROJECT_ROOT/examples/flowDet/train_onlyFlow_yolo.sh" "lru1_drone"
bash "$PROJECT_ROOT/examples/flowDet/train_onlyFlow_yolo.sh" "lru1_lander"
bash "$PROJECT_ROOT/examples/flowDet/train_onlyFlow_yolo.sh" "lru1_lru2"
