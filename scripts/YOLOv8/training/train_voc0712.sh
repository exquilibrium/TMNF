#!/bin/sh

# Sometimes slurm just fails, try running a few times

nvidia-smi

#yolo stats "/media/chen/76AECF8EAECF4579/data/VOCdevkit_xml/VOC0712/data_CS.yaml"
yolo detect train model=yolov8n.pt data="/media/chen/76AECF8EAECF4579/data/VOCdevkit_xml/VOC0712/data_CS.yaml" epochs=50 imgsz=640 amp=False
