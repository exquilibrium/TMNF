#!/bin/bash



# First 15 classes of PASCAL-VOC
# === SET PATH TO DATASET DIRECTORY!!! ===
MODEL_PATH="/home/chen/openset_detection/scripts/YOLOv8/training/runs/detect/train_voc0712/weights/best.pt"
TRAIN_SET="/media/chen/76AECF8EAECF4579/data/VOCdevkit_xml/VOC0712/ImageSets/YOLO_CS/train.txt"
VAL_SET="/media/chen/76AECF8EAECF4579/data/VOCdevkit_xml/VOC0712/ImageSets/YOLO_CS/val.txt"
TEST_SET="/media/chen/76AECF8EAECF4579/data/VOCdevkit_xml/VOC0712/ImageSets/YOLO/test.txt"
CONF_THRESH=0.2
IOU_THRESH=0.5
# === SET PATH TO DATASET DIRECTORY!!! ===
DS="xml"
saveNm="FlowDet_Voc_clsLogits_${DS}_yolo" # FlowDet_Voc_clsLogits_xml_yolo



# Path to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Extract features
python3 "$SCRIPT_DIR/feat_extraction_yolo.py" "$MODEL_PATH" "$TRAIN_SET" "$saveNm" --num_classes 15 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH
python3 "$SCRIPT_DIR/feat_extraction_yolo.py" "$MODEL_PATH" "$VAL_SET" "$saveNm" --num_classes 15 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH
python3 "$SCRIPT_DIR/feat_extraction_yolo.py" "$MODEL_PATH" "$TEST_SET" "$saveNm" --num_classes 15 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH

# Assign predictions
python3 "$SCRIPT_DIR/pred_assignment_yolo.py" "$MODEL_PATH" "$TRAIN_SET" "$saveNm" --num_classes 15 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH
python3 "$SCRIPT_DIR/pred_assignment_yolo.py" "$MODEL_PATH" "$VAL_SET" "$saveNm" --num_classes 15 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH
python3 "$SCRIPT_DIR/pred_assignment_yolo.py" "$MODEL_PATH" "$TEST_SET" "$saveNm" --num_classes 15 --conf_thresh $CONF_THRESH --iou_thresh $IOU_THRESH
