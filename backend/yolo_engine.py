# yolo_engine.py
import tempfile, os
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from ultralytics import YOLO

_MODEL_CACHE = {}

def load_yolo_model(weights_path: str):
    global _MODEL_CACHE
    if weights_path in _MODEL_CACHE:
        return _MODEL_CACHE[weights_path]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(weights_path)
    try: model.fuse()
    except: pass

    _MODEL_CACHE[weights_path] = (model, device)
    return model, device

def infer_with_embedding(model, device, img_path, imgsz=1024, conf=0.25):
    res = model.predict(
        source=img_path,
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=0.45,
        verbose=False
    )[0]

    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
    confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else []
    masks = res.masks.data.cpu().numpy() if res.masks is not None else []

    dets = []
    for i, box in enumerate(boxes):
        dets.append({
            "box": tuple(float(x) for x in box),
            "confidence": float(confs[i]),
            "mask": masks[i] if len(masks) > i else None
        })

    # embedding extraction (simplified)
    embedding = []
    try:
        feats = res.feats
        if feats is not None:
            t = feats[-1] if isinstance(feats, (list,tuple)) else feats
            embedding = t.cpu().numpy().flatten().tolist()
    except:
        pass

    return dets, embedding
