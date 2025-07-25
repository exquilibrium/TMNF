import json
import os
from pathlib import Path
import argparse

import cv2
import numpy as np
import torch
import tqdm

from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import ops

# https://gist.github.com/justinkay/8b00a451b1c1cc3bcf210c86ac511f46
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_results_json(results, save_path):
    json_str = json.dumps(results, cls=NumpyEncoder)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(f"{save_path}.json", 'w') as f:
        f.write(json_str)
    print(f"Saved to {save_path}.json")

class SaveIO:
    """Simple PyTorch hook to save the output of a nn.module."""
    def __init__(self):
        self.input = None
        self.output = None
        
    def __call__(self, module, module_in, module_out):
        self.input = module_in
        self.output = module_out

def load_and_prepare_model(model_path):
    """
    Load YOLO model and register forward hooks.
    Hook intermediate features and raw predictions to reverse-engineer YOLO model outputs.

    Args:
        model_path (YOLO): Path to YOLOv8 model

    Returns:
        model: YOLO model.
        hooks: List of registered hook references
    """
    # Load YOLO model
    model = YOLO(model_path)
    detect = None
    cv2_hooks = None
    cv3_hooks = None
    detect_hook = SaveIO()

    # Identify Detect layer (YOLO head) and register forward hook (`detect_hook`)
    for i, module in enumerate(model.model.modules()):
        if type(module) is Detect:
            module.register_forward_hook(detect_hook)
            detect = module

            # Register forward hooks on detection scale's internal convolution layers (`cv2` and `cv3`)
            cv2_hooks = [SaveIO() for _ in range(module.nl)]
            cv3_hooks = [SaveIO() for _ in range(module.nl)]
            for i in range(module.nl):
                module.cv2[i].register_forward_hook(cv2_hooks[i])
                module.cv3[i].register_forward_hook(cv3_hooks[i])
            break
    input_hook = SaveIO()

    # Register top-level forward hook on entire model
    model.model.register_forward_hook(input_hook)
    hooks = [input_hook, detect, detect_hook, cv2_hooks, cv3_hooks]

    return model, hooks

from torch.utils.data import Dataset, DataLoader

class ImagePathDataset(Dataset):
    def __init__(self, image_set_txt):
        with open(image_set_txt, 'r') as f:
            self.image_paths = [Path(line.strip()) for line in f if line.strip().endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(str(path))
        return path, img

def run_predict(img, img_path, model, hooks, num_classes, conf_threshold, iou_threshold):
    """
    Run prediction with a YOLO model and apply Non-Maximum Suppression (NMS) to the results.

    Args:
        img_path (str): Path to an image file.
        model (YOLO): YOLO model object.
        hooks (list): List of hooks for the model.
        num_classes (int): Number of classes.
        conf_threshold (float, optional): Confidence threshold for detection. Default is 0.5.
        iou_threshold (float, optional): Intersection over Union (IoU) threshold for NMS. Default is 0.7.

    Returns:
        list: List of selected bounding box dictionaries after NMS.
    """
    # Unpack hooks from load_and_prepare_model()
    input_hook, detect, detect_hook, cv2_hooks, cv3_hooks = hooks

    # Run inference. Results are stored by hooks
    with torch.no_grad():
        model(imgs_batch, verbose=False, half=True)

    # Reverse engineer outputs to find logits
    # See Detect.forward(): https://github.com/ultralytics/ultralytics/blob/b638c4ed9a24270a6875cdd47d9eeda99204ef5a/ultralytics/nn/modules/head.py#L22
    shape = detect_hook.input[0][0].shape  # BCHW
    x = []
    for i in range(detect.nl):
        x.append(torch.cat((cv2_hooks[i].output, cv3_hooks[i].output), 1))
    x_cat = torch.cat([xi.view(shape[0], detect.no, -1) for xi in x], 2)
    box, classes = x_cat.split((detect.reg_max * 4, detect.nc), 1)

    # Assume batch size 1 (i.e. just running with one image)
    # Loop here to batch images
    batch_idx = 0
    xywh_sigmoid = detect_hook.output[0][batch_idx]
    all_logits = classes[batch_idx]

    # Get original image shape and model image shape to transform boxes
    img_shape = input_hook.input[0].shape[2:]
    orig_img_shape = model.predictor.batch[1][batch_idx].shape[:2]

    # Transpose data
    xywh_sigmoid = xywh_sigmoid.T      # shape: [6300, 4 + C]
    all_logits = all_logits.T          # shape: [6300, C]

    # Compute predictions
    # Slice tensors
    coords = xywh_sigmoid[:, :4]          # [N, 4] — x0, y0, x1, y1
    activations = xywh_sigmoid[:, 4:]     # [N, C]
    logits = all_logits                   # [N, C]

    # Scale all boxes using vector ops (no loop)
    coords_cpu = coords.detach().cpu().numpy()  # shape [N, 4]
    scaled_coords = np.array([
        yolo_ops.scale_boxes(img_shape, coords_cpu[i], orig_img_shape)
        for i in range(coords_cpu.shape[0])
    ])  # shape [N, 4]

    # Convert to cx, cy, w, h (bbox_xywh)
    scaled_coords = torch.tensor(scaled_coords, dtype=torch.float32)  # [N, 4]
    x0, y0, x1, y1 = scaled_coords[:, 0], scaled_coords[:, 1], scaled_coords[:, 2], scaled_coords[:, 3]
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    w = x1 - x0
    h = y1 - y0
    bbox_xywh = torch.stack([cx, cy, w, h], dim=1)  # [N, 4]
    
    # Non-Maximum-Suppresion (Retain only the most relevant boxes based on confidence scores)
    boxes_for_nms = torch.cat([
        bbox_xywh.to(coords.device)  ,       # [N, 4]
        activations.to(coords.device)  ,     # [N, C]
        activations.to(coords.device)  ,     # [N, C] again (YOLO-style input)
        logits.to(coords.device)             # [N, C]
    ], dim=1).T.unsqueeze(0)  # Transpose to final shape: [1, 4+3C, N] -> [1, N, 4+3C]

    nms_results_batch = yolo_ops.non_max_suppression(
        boxes_for_nms,
        conf_thres=conf_threshold,
        iou_thres=iou_threshold,
        nc=detect.nc
    )

    # Initialize final output: one list per class
    class_results = [[] for _ in range(num_classes)]
    for nms_results in nms_results_batch:
        if nms_results is None or nms_results.shape[0] == 0:
            continue

        for b in range(nms_results.shape[0]):
            box = nms_results[b, :]
            x0, y0, x1, y1, conf, cls, *acts_and_logits = box
            logits = acts_and_logits[detect.nc:]
            cls_idx = int(cls.item())
            bbox = [x0.item(), y0.item(), x1.item(), y1.item()]  # xyxy
            score = conf.item()
            logits = [p.item() for p in logits]


            # Pack: [x1, y1, x2, y2, score, logit_0, ..., logit_C]
            row = bbox + [score] + logits
            class_results[cls_idx].append(row)

    # Convert each class list to np.ndarray of shape (k_i, 5+num_classes)
    class_results = [
        np.array(cls_boxes, dtype=np.float32) if cls_boxes else np.zeros((0, 5 + num_classes), dtype=np.float32)
        for cls_boxes in class_results
    ]

    return class_results

def run_inference_yolo(model_path, image_set, num_classes, confThresh, iouThresh):
    """
    Process raw YOLO output from images in a folder using captured logits.
    Collects detections per image efficiently using vectorized operations.

    Args:
        model: Trained Ultralytics YOLO model.
        image_set (Path): Path to imageset txt.
        num_classes (int): Number of classes and logits.
        confThresh (float): Confidence threshold for keeping detections.
        iou_threshold (float): IOU threshold for Non-Maximum Suppression (NMS).

    Returns:
        dict: {image_name: list of detections}, each detection = [x1, y1, x2, y2, score, class_idx]
    """
    maxOneDetOneRP = True
    # Batching not implemented
    batch_size = 1
    # Preload images (keep original path and loaded image)
    dataset = ImagePathDataset(args.imageset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Load model
    model, hooks = load_and_prepare_model(model_path)

    # Main loop
    for image_paths_batch, imgs_batch in tqdm.tqdm(dataloader):
        image_path = image_paths_batch[0]
        img = imgs_batch[0]

        imgName = image_path.name
        allResults[imgName] = []
        all_detections = []

        # Predict logits using image and path
        results = run_predict(img, image_path, model, hooks, num_classes, conf_threshold=confThresh, iou_threshold=iouThresh)

        for cls_idx in range(num_classes):
            imDets = results[cls_idx]
            if len(imDets) > 0:
                logits = imDets[:, 5:5 + num_classes]
                scores = imDets[:, 4]
                mask = None
                if maxOneDetOneRP:
                    mask = np.argmax(logits, axis=1) == cls_idx
                    if np.sum(mask) > 0:
                        imDets = imDets[mask]
                        scores = scores[mask]
                    else:
                        continue

                if confThresh > 0.:
                    mask = scores >= confThresh
                    if np.sum(mask) > 0:
                        imDets = imDets[mask]
                    else:
                        continue

                all_detections.append(imDets)

        if len(all_detections) == 0:
            continue
        else:
            all_detections = np.concatenate(all_detections, axis=0)
            detections, _ = np.unique(all_detections, return_index=True, axis=0)

        allResults[imgName] = detections.tolist()

    return allResults

# CLI
def main():
    parser = argparse.ArgumentParser(description='Extract logits for train, val, test set.')
    parser.add_argument('model_path', type=str, help='Path to YOLO model.')
    parser.add_argument('image_set', type=str, help='Path to imageset txt.')
    parser.add_argument('saveNm', type=str, help='Save name of output')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes.')
    parser.add_argument('--conf_thresh', type=float, default=0.2, help='Confidence Threshold.')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IOU Threshold.')
    args = parser.parse_args()

    print('Extracting features!')
    results = run_inference_yolo(args.model_path,
                            args.image_set,
                            num_classes=args.num_classes,
                            confThresh=args.conf_thresh,
                            iouThresh=args.iou_thresh)
    # Setting path
    split = Path(args.image_set).stem # train/val/test
    parts = args.model_path.split('/') # /home/chen/openset_detection/.../best.pt
    base = f"/{parts[1]}/{parts[2]}/TMNF" # /home/chen/TMNF
    save_path = f'{base}/data/datasets/extracted_feat/flowDet/YOLOv8/raw/XMLDataset/{split}/{args.saveNm}'
    save_results_json(results, save_path)


if __name__ == "__main__":
    main()
