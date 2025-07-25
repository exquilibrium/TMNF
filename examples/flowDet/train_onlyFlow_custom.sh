#!/bin/bash 
path=$( dirname -- "$( readlink -f -- "$0"; )"; )

CONFIG="rnvp_logits_rbf_cls_ib" # [rnvp,nsf,residual]_logits_["", gmm, gmm_cls, gmm_cls_ib, rbf, rbf_cls, rbf_cls_ib]
SCRIPT="train_onlyFlow.py"

NF_CONFIF="flowDet/train/onlyFlow_voc/oF_frcnn_voc_rnvp_logits_rbf_cls_ib_${1}.py"
FEAT_FN="FlowDet_Voc_clsLogits_${1}" # "GMMDet_Voc_msFeats", "CE_Voc_msFeats"

FEAT_TYPE="logits" # 'logits'
ONLY_EVAL=True

cd $path
echo 'In directory:'$path

if [ $ONLY_EVAL = "True" ]
then
    echo "Only testing NF for $CONFIG ..."
    python $SCRIPT --config $NF_CONFIF --feat_fn $FEAT_FN --feat_type $FEAT_TYPE --only_eval 
else
    echo "Training and testing NF  for $CONFIG ..."
    python $SCRIPT --config $NF_CONFIF --feat_fn $FEAT_FN --feat_type $FEAT_TYPE
fi
