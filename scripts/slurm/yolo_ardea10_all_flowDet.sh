#!/bin/bash

# Base project root (2 levels up from slurm script)
#HOME_ROOT="/home/chen/TMNF"
HOME_ROOT="/home/chen_le"

#DATASET_ROOT="/media/chen/76AECF8EAECF4579/data"
DATASET_ROOT="/volume/hot_storage/slurm_data/chen_le/ARCHES"

PROJECT_ROOT="${HOME_ROOT}/TMNF"

# Run extract script
bash "$PROJECT_ROOT/scripts/YOLOv8/extraction/extract_ardea10.sh" $HOME_ROOT $DATASET_ROOT
bash "$PROJECT_ROOT/scripts/YOLOv8/extraction/extract_ardea10_lander.sh" $HOME_ROOT $DATASET_ROOT
bash "$PROJECT_ROOT/scripts/YOLOv8/extraction/extract_ardea10_lru1.sh" $HOME_ROOT $DATASET_ROOT
bash "$PROJECT_ROOT/scripts/YOLOv8/extraction/extract_ardea10_lru2.sh" $HOME_ROOT $DATASET_ROOT

# Run train and test flowDet script
#bash "$PROJECT_ROOT/examples/flowDet/train_onlyFlow_yolo.sh" "ardea10"
#bash "$PROJECT_ROOT/examples/flowDet/train_onlyFlow_yolo.sh" "ardea10_lander"
#bash "$PROJECT_ROOT/examples/flowDet/train_onlyFlow_yolo.sh" "ardea10_lru1"
#bash "$PROJECT_ROOT/examples/flowDet/train_onlyFlow_yolo.sh" "ardea10_lru2"
