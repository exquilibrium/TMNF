#!/bin/sh

# Sometimes slurm just fails, try running a few times

nvidia-smi

#yolo detect train model=yolov8n.pt data="/volume/hot_storage/slurm_data/chen_le/ARCHES/ardea10_run04_labels/data.yaml" epochs=50 imgsz=640 amp=False
#yolo detect train model=yolov8n.pt data="/volume/hot_storage/slurm_data/chen_le/ARCHES/ardea10_run04_labels/data_CS_lru2.yaml" epochs=50 imgsz=640 amp=False
yolo detect train model=yolov8n.pt data="/media/chen/76AECF8EAECF4579/data/ardea10_run04_labels/data_CS_lru2.yaml" epochs=50 imgsz=640 amp=False
