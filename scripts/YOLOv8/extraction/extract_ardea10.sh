#!/bin/bash



# First 15 classes of PASCAL-VOC
# === SET PATH TO DATASET DIRECTORY!!! ===
#MODEL_BASE_DIR="/home/chen"
MODEL_BASE_DIR="/home/chen_le"
#DATASET_BASE_DIR="/media/chen/76AECF8EAECF4579/data"
DATASET_BASE_DIR="/volume_hot_storage/slurm_data/chen_le/ARCHES"
DS="ardea10"

MODEL_PATH="${MODEL_BASE_DIR}/openset_detection/scripts/YOLOv8/training/runs/detect/train_${DS}/weights/best.pt"
TRAIN_SET="${DATASET_BASE_DIR}/${DS}_all/ImageSets/YOLO/train.txt"
VAL_SET="${DATASET_BASE_DIR}/${DS}_all/ImageSets/YOLO/val.txt"
TEST_SET="${DATASET_BASE_DIR}/${DS}_all/ImageSets/YOLO/test.txt"
CONF_THRESH=0.2
IOU_THRESH=0.5
# === SET PATH TO DATASET DIRECTORY!!! ===
saveNm="FlowDet_Voc_clsLogits_${DS}_yolo" # FlowDet_Voc_clsLogits_lru1_yolo



# Path to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Extract features
python3 "$SCRIPT_DIR/feat_extraction_yolo.py" "$MODEL_PATH" "$TRAIN_SET" "$saveNm" --num_classes 3 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH
python3 "$SCRIPT_DIR/feat_extraction_yolo.py" "$MODEL_PATH" "$VAL_SET" "$saveNm" --num_classes 3 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH
python3 "$SCRIPT_DIR/feat_extraction_yolo.py" "$MODEL_PATH" "$TEST_SET" "$saveNm" --num_classes 3 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH

# Assign predictions
python3 "$SCRIPT_DIR/pred_assignment_yolo.py" "$MODEL_PATH" "$TRAIN_SET" "$saveNm" --num_classes 3 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH
python3 "$SCRIPT_DIR/pred_assignment_yolo.py" "$MODEL_PATH" "$VAL_SET" "$saveNm" --num_classes 3 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH
python3 "$SCRIPT_DIR/pred_assignment_yolo.py" "$MODEL_PATH" "$TEST_SET" "$saveNm" --num_classes 3 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH
