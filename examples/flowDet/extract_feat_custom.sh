#!/bin/bash 
path=$( dirname -- "$( readlink -f -- "$0"; )"; )

CONFIG="flowDet/feat_ext/faster_rcnn_r50_fpn_voc0712OS_clsLogits_${1}.py"
WEIGHT_DIR="/home/chen/openset_detection/mmdetection/weights/frcnnCEwAnchorVocCS_${1}"
SAVE_NAME="FlowDet_Voc_clsLogits_${1}" # GMMDet_Voc_msFeats, CE_Voc_msFeats, FlowDet_Voc_msFeats


DATASPLIT_TYPE="flowDet" #  "GMMDet" or "flowDet"
CONF_THR=0.2 # 0.2 in GMM paper
IOU_THR=0.5  # 0.5 in GMM paper
maxOneDetOneRP=True # default is True
CKP="latest.pth" # latest.pth

cd $path
echo 'In directory:'$path\

CUSTOM="True"

if [ $CUSTOM == "True" ]
then
    ########################################## Feature Extraction ##########################################
    ext_feat_cmd="python feat_extraction_custom.py --checkpoint $CKP --datasplit_type $DATASPLIT_TYPE --config $CONFIG --weights_dir $WEIGHT_DIR --saveNm $SAVE_NAME --confThresh $CONF_THR --maxOneDetOneRP $maxOneDetOneRP"
    echo "Extracting features from training set"
    cmd=$ext_feat_cmd" --subset train "
    echo "executing $cmd"
    $cmd

    echo "Extracting features from VAL set"
    cmd=$ext_feat_cmd" --subset val "
    echo "executing $cmd"
    $cmd

    echo "Extracting features from TEST set"
    cmd=$ext_feat_cmd" --subset testOS "
    echo "executing $cmd"
    $cmd

    ########################################## Prediction Assignment ##########################################
    ass_cmd="python pred_assignment_custom.py --datasplit_type $DATASPLIT_TYPE --config $CONFIG --confThresh $CONF_THR --iouThresh $IOU_THR --saveNm $SAVE_NAME"
    echo "assigning types to training set"
    cmd=$ass_cmd" --subset train "
    echo "executing $cmd"
    $cmd

    echo "assigning types to val set"
    cmd=$ass_cmd" --subset val "
    echo "executing $cmd"
    $cmd

    echo "assigning types to test set"
    cmd=$ass_cmd" --subset test "
    echo "executing $cmd"
    $cmd
else
    ########################################## Feature Extraction ##########################################
    ext_feat_cmd="python feat_extraction.py --checkpoint $CKP --datasplit_type $DATASPLIT_TYPE --config $CONFIG --weights_dir $WEIGHT_DIR --saveNm $SAVE_NAME --confThresh $CONF_THR --maxOneDetOneRP $maxOneDetOneRP"
    echo "Extracting features from training set"
    cmd=$ext_feat_cmd" --subset train "
    echo "executing $cmd"
    $cmd

    echo "Extracting features from VAL set"
    cmd=$ext_feat_cmd" --subset val "
    echo "executing $cmd"
    $cmd

    echo "Extracting features from TEST set"
    cmd=$ext_feat_cmd" --subset testOS "
    echo "executing $cmd"
    $cmd

    ########################################## Prediction Assignment ##########################################
    ass_cmd="python pred_assignment.py --datasplit_type $DATASPLIT_TYPE --config $CONFIG --confThresh $CONF_THR --iouThresh $IOU_THR --saveNm $SAVE_NAME"
    echo "assigning types to training set"
    cmd=$ass_cmd" --subset train "
    echo "executing $cmd"
    $cmd

    echo "assigning types to val set"
    cmd=$ass_cmd" --subset val "
    echo "executing $cmd"
    $cmd

    echo "assigning types to test set"
    cmd=$ass_cmd" --subset test "
    echo "executing $cmd"
    $cmd
fi