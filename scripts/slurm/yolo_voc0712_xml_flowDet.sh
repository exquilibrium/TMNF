#!/bin/bash

# Base project root (2 levels up from slurm script)
#HOME_ROOT="/home/chen/TMNF"
HOME_ROOT="/home/chen_le"

#DATASET_ROOT="/media/chen/76AECF8EAECF4579/data"
DATASET_ROOT="/volume/hot_storage/slurm_data/chen_le"

PROJECT_ROOT="${HOME_ROOT}/TMNF"

# Run extract script
bash "$PROJECT_ROOT/scripts/YOLOv8/extraction/extract_voc0712.sh" $HOME_ROOT $DATASET_ROOT

# Run train and test flowDet script
#bash "$PROJECT_ROOT/examples/flowDet/train_onlyFlow_yolo.sh" "xml"
