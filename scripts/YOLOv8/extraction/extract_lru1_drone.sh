#!/bin/bash



# First 15 classes of PASCAL-VOC
# === SET PATH TO DATASET DIRECTORY!!! ===
MODEL_BASE_DIR="/home/chen"
DATASET_BASE_DIR="/media/chen/76AECF8EAECF4579/data"
CS="drone"

MODEL_PATH="${MODEL_BASE_DIR}/openset_detection/scripts/YOLOv8/training/runs/detect/train_lru1_${CS}/weights/best.pt"
TRAIN_SET="${DATASET_BASE_DIR}/lru1_all/ImageSets/YOLO_CS_${CS}/train.txt"
VAL_SET="${DATASET_BASE_DIR}/lru1_all/ImageSets/YOLO_CS_${CS}/val.txt"
TEST_SET="${DATASET_BASE_DIR}/lru1_all/ImageSets/YOLO/test.txt"
CONF_THRESH=0.2
IOU_THRESH=0.5
# === SET PATH TO DATASET DIRECTORY!!! ===
DS="lru1"
saveNm="FlowDet_Voc_clsLogits_${DS}_${CS}_yolo" # FlowDet_Voc_clsLogits_lru1_drone_yolo



# Path to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Extract features
python3 "$SCRIPT_DIR/feat_extraction_yolo.py" "$MODEL_PATH" "$TRAIN_SET" "$saveNm" --num_classes 2 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH
python3 "$SCRIPT_DIR/feat_extraction_yolo.py" "$MODEL_PATH" "$VAL_SET" "$saveNm" --num_classes 2 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH
python3 "$SCRIPT_DIR/feat_extraction_yolo.py" "$MODEL_PATH" "$TEST_SET" "$saveNm" --num_classes 2 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH

# Assign predictions
python3 "$SCRIPT_DIR/pred_assignment_yolo.py" "$MODEL_PATH" "$TRAIN_SET" "$saveNm" --num_classes 2 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH
python3 "$SCRIPT_DIR/pred_assignment_yolo.py" "$MODEL_PATH" "$VAL_SET" "$saveNm" --num_classes 2 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH
python3 "$SCRIPT_DIR/pred_assignment_yolo.py" "$MODEL_PATH" "$TEST_SET" "$saveNm" --num_classes 2 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH
