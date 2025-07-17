import sys
import os
import argparse
import numpy as np
from mmcv import Config
from pathlib import Path
import importlib

from gmmDet_utils_yolov8 import *

# Import CONFIG by adding TMNF to path
tmnf_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, tmnf_root)

# Functions
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_results_json(results, save_path):
    jsonRes = json.dumps(results, cls=NumpyEncoder)
    save_dir = os.path.split(save_path)[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    f = open(f"{save_path}.json", 'w')
    f.write(jsonRes)
    print(f"Saving to {save_path}.json")
    f.close()

def pred_assignment(datasplit: str,
                    num_classes: int,
                    confThresh: float,
                    iouThresh: float,
                    base_path: str,
                    saveNm: str):
    dataset_type = 'XMLDataset'
    saveFileName = saveNm
    save_dir = f"{base_path}/data/datasets/extracted_feat/flowDet/YOLOv8/associated/{dataset_type}"
    raw_res_dir = f"{base_path}/data/datasets/extracted_feat/flowDet/YOLOv8/raw/XMLDataset/{datasplit}"
    
    # process train data
    if datasplit == "train":
        ds_subsets = [data['trainCS']]
        raw_res_pth_list = [f"{raw_res_dir}/{saveNm}.json"]

        print(f'Assigning train data on {dataset_type}')
        print(f"Loading outputs from {raw_res_pth_list}.")

        # Train on CLOSED-SET
        trainDict = assign_train(ds_subsets, raw_res_pth_list, num_classes, iouThresh)

        print(f"detScores: {len(trainDict['scores'])} with avg {np.mean(trainDict['scores'])}")
        print(f"detLogits: {len(trainDict['logits'])}")
        print(f"all_filenames: {len(trainDict['filenames'])}")
        save_results_json(trainDict, os.path.join(save_dir, f'train/{saveFileName}'))
        
    # process eval data
    elif datasplit == "val":
        print(f'Assigning val data on {dataset_type}.')
        raw_output_file =  f"{raw_res_dir}/{saveNm}.json"
        print(f"Loading outputs from {raw_output_file}.")

        # Val on CLOSED-SET
        evalDict = assign_eval(data['val'], raw_output_file, num_classes, confThresh, iouThresh)

        print(f"detScores: {len(evalDict['scores'])} with avg {np.mean(evalDict['scores'])}")
        print(f"detLogits: {len(evalDict['logits'])}")
        print(f"type: {len(evalDict['type'])}")
        print(f"detFeats: {len(evalDict['feats'])}")
        print(f"all_filenames: {len(evalDict['filenames'])}")
        print(f"#type 0: {np.sum(np.array(evalDict['type'])==0)}")
        print(f"#type 1: {np.sum(np.array(evalDict['type'])==1)}")
        print(f"#type 2: {np.sum(np.array(evalDict['type'])==2)}")
        save_results_json(evalDict, os.path.join(save_dir, f'val/{saveFileName}'))
    
    elif datasplit == "test":
        gtLabel_mapping = None

        evalDict = {}
        print(f'Assigning Open-Set test data on {dataset_type}.')
        raw_output_file =  f"{raw_res_dir}/{saveNm}.json"
        print(f"Loading outputs from {raw_output_file}.")

        # Test on OPEN-SET
        evalDict = assign_eval(data['testOS'], raw_output_file, num_classes, confThresh, iouThresh, gtLabel_mapping=gtLabel_mapping)

        print(f"detScores: {len(evalDict['scores'])} with avg {np.mean(evalDict['scores'])}")
        print(f"detScores for type 2 avg: {np.mean(np.array(evalDict['scores'])[np.array(evalDict['type'])==2])}")
        print(f"detScores for type 1 avg: {np.mean(np.array(evalDict['scores'])[np.array(evalDict['type'])==1])}")
        print(f"detLogits: {len(evalDict['logits'])}")
        print(f"type: {len(evalDict['type'])}")
        print(f"detFeats: {len(evalDict['feats'])}")
        print(f"all_filenames: {len(evalDict['filenames'])}")
        print(f"#type 0: {np.sum(np.array(evalDict['type'])==0)}")
        print(f"#type 1: {np.sum(np.array(evalDict['type'])==1)}")
        print(f"#type 2: {np.sum(np.array(evalDict['type'])==2)}")
        save_results_json(evalDict, os.path.join(save_dir, f'test/{saveFileName}'))

    else:
        print("Invalid datasplit")

# CLI: TODO
def main():
    parser = argparse.ArgumentParser(description='Assign predictions for train, val, test set.')
    parser.add_argument('model_path', type=str, help='Path to YOLO model.')
    parser.add_argument('image_set', type=str, help='Split: train/val/test. Note: train and val are CS, but test is OS.')
    parser.add_argument('saveNm', type=str, help='Save name of output')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes.')
    parser.add_argument('--conf_thresh', type=float, default=0.2, help='Confidence Threshold.')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IOU Threshold.')
    args = parser.parse_args()

    print('Assigning Predictions!')

    # Config loader
    prefix = "FlowDet_Voc_clsLogits_"
    suffix = "_yolo"
    config_map = {
        "ardea10": "voc0712OS_ardea10",
        "ardea10_lander": "voc0712OS_ardea10_lander",
        "ardea10_lru1": "voc0712OS_ardea10_lru1",
        "ardea10_lru2": "voc0712OS_ardea10_lru2",
        "lru1": "voc0712OS_lru1",
        "lru1_drone": "voc0712OS_lru1_drone",
        "lru1_lander": "voc0712OS_lru1_lander",
        "lru1_lru2": "voc0712OS_lru1_lru2",
        "xml": "voc0712OS_xml"
    }
    config_key = args.saveNm[len(prefix):-len(suffix)]
    module_name = config_map[config_key]
    config_module = importlib.import_module(f"configs_imported.{module_name}")
    # Equivalent to: from configs_imported.module_name import *
    globals().update(vars(config_module))
    print(f"Imported dataset config: configs_imported.{module_name}")

    # Setting path
    split = Path(args.image_set).stem # Split: train/val/test.
    parts = args.model_path.split('/') # /home/chen/openset_detection/.../best.pt
    base = f"/{parts[1]}/{parts[2]}/TMNF" # /home/chen/TMNF


    pred_assignment(split, args.num_classes, args.conf_thresh, args.iou_thresh, base, args.saveNm)
        

if __name__ == "__main__":
    main()

