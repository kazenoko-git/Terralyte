#!/usr/bin/env python3
"""
run_model.py - Terralyte AI Analysis (SolarDetection-compatible output)
- Extracts YOLO embeddings, aggregates detections, calls QC model which consumes embeddings.
- Outputs judge JSON (single object) to stdout and appends to qc_training_data.json (including embedding).
"""

import os
os.environ["YOLO_VERBOSE"] = "False"
os.environ["ULTRALYTICS_HUB"] = "False"
os.environ["ULTRALYTICS_LOGGING"] = "False"
import warnings
warnings.filterwarnings("ignore")
import json
import math
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import sys
def silent(*a, **k): pass

try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

from ultralytics import YOLO

# QC inference model
from inference import Model, generateQcNotes

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path("/Users/ivansamuel/PycharmProjects/Terralyte")
TRAINING_JSON_PATH = PROJECT_ROOT / "qc_training_data.json"
MODEL_PICKLE_PATH = PROJECT_ROOT / "solarQcModel.pkl"
AUDIT_DIR = PROJECT_ROOT / "audit_overlays"
AUDIT_DIR.mkdir(exist_ok=True, parents=True)

DEFAULT_IMG_SIZE = 1024
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_ZOOM = 18
DEFAULT_TILE_PX = 256  # typical web-mercator tile size

PANEL_POWER_DENSITY = 180.0   # W/m²
AVG_PANEL_AREA_SQM = 1.7      # typical area of 1 solar panel (m²) — tweak if you want

# ============================================================
# HELPERS
# ============================================================
def nowStr():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def shaDetection(imagePath: str, summary: Dict):
    s = f"{imagePath}|{summary.get('panel_count',0)}|{summary.get('area_sqm',0)}|{summary.get('capacity_kw',0)}"
    return hashlib.sha256(s.encode()).hexdigest()

def metersPerPixel(lat_deg: float, zoom: int):
    # meters per pixel at given zoom assuming 256px tile (Web Mercator)
    return 156543.03392 * math.cos(math.radians(lat_deg)) / (2 ** zoom)

# ============================================================
# PANEL METRICS (improved MPP scaling)
# ============================================================
def _effective_mpp(lat, zoom, img_width_px):
    """
    Estimate meters-per-pixel for the given stitched image width.
    If width is an integer multiple of 256 (N tiles), scale base MPP accordingly;
    else approximate using tile-count = round(width/256).
    """
    tiles_side = max(1, int(round(img_width_px / DEFAULT_TILE_PX)))
    base_mpp = metersPerPixel(lat, zoom)  # meters/pixel assuming 256px tile
    # If img was stitched from N tiles then image_pixel_for_tile = img_width_px / tiles_side
    # scale = (256 / (img_width_px / tiles_side)) = 256 * tiles_side / img_width_px
    scale = (DEFAULT_TILE_PX * tiles_side) / float(max(1, img_width_px))
    eff = base_mpp * scale
    return eff

def estimateFromMasks(masks, width, height, lat=None, zoom=DEFAULT_ZOOM):
    if masks is None:
        return 0.0, 0.0

    masks_np = np.asarray(masks)
    if masks_np.ndim == 2:
        masks_np = masks_np[None]

    if masks_np.size == 0:
        return 0.0, 0.0

    pixel_count = masks_np.reshape(len(masks_np), -1).sum()

    if lat is None:
        lat = 0.0

    mpp = _effective_mpp(lat, zoom, width)
    sqm = pixel_count * (mpp ** 2)
    kw = (sqm * PANEL_POWER_DENSITY / 1000.0) * 0.85
    return float(sqm), float(min(max(kw, 0.0), 20000.0))

def estimateFromBoxes(boxes, width, height, lat=None, zoom=DEFAULT_ZOOM):
    if not boxes:
        return 0.0, 0.0

    if lat is None:
        lat = 0.0

    mpp = _effective_mpp(lat, zoom, width)

    total_pixels = 0.0
    for (x1, y1, x2, y2) in boxes:
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        total_pixels += w * h

    sqm = total_pixels * (mpp ** 2)
    kw = (sqm * PANEL_POWER_DENSITY / 1000.0) * 0.75
    return float(sqm), float(min(max(kw, 0.0), 20000.0))

# ============================================================
# OVERLAY
# ============================================================
def saveOverlay(imagePath, masks, boxes, outPath, alpha=0.35):
    try:
        img = Image.open(imagePath).convert("RGBA")
    except Exception:
        return None

    canvas = Image.new("RGBA", img.size)
    draw = ImageDraw.Draw(canvas, "RGBA")

    if masks is not None:
        masks_np = np.asarray(masks)
        if masks_np.ndim == 2:
            masks_np = masks_np[None]
        for m in masks_np:
            try:
                mk = Image.fromarray((m * 255).astype("uint8"), "L").resize(img.size, Image.NEAREST)
                col = Image.new("RGBA", img.size, (0, 200, 100, int(255 * alpha)))
                canvas.paste(col, (0,0), mk)
            except Exception:
                pass

    for (x1, y1, x2, y2) in boxes:
        draw.rectangle([x1,y1,x2,y2], outline=(255,255,0,255), width=2)

    out = Image.alpha_composite(img, canvas)
    out = out.filter(ImageFilter.SHARPEN)
    out.save(outPath)
    return str(outPath)

# ============================================================
# TRAINING JSON append (adds embedding)
# ============================================================
def appendTraining(summary: Dict, imagePath: str):
    TRAINING_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    if TRAINING_JSON_PATH.exists():
        try:
            data = json.load(open(TRAINING_JSON_PATH))
        except Exception:
            data = []
    else:
        data = []

    # dedupe by hash of detection summary
    fp = shaDetection(imagePath, summary)
    for item in data:
        if shaDetection(item.get("image_path",""), item) == fp:
            return False

    entry = {
        "image_path": imagePath,
        "timestamp": nowStr(),
        "confidence": float(summary.get("confidence", 0.0)),
        "panel_count": int(summary.get("panel_count", 0)),
        "area_sqm": float(summary.get("area_sqm", 0.0)),
        "capacity_kw": float(summary.get("capacity_kw", 0.0)),
        "has_solar": bool(summary.get("has_solar", False)),
        "qc_notes": summary.get("qc_notes", []),
        "embedding": summary.get("embedding", [])
    }

    data.append(entry)
    json.dump(data, open(TRAINING_JSON_PATH, "w"), indent=2)
    return True

# ============================================================
# AGGREGATION + CLUSTERING
# ============================================================
def _centroid_from_box(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def cluster_centroids(centroids, mpp, distance_m=10.0):
    """
    Simple single-linkage clustering in pixel coords using threshold in meters.
    - centroids: list of (x_px, y_px)
    - mpp: meters per pixel
    - distance_m: cluster threshold (meters)
    Returns: list of cluster indices for each centroid
    """
    if not centroids:
        return []

    pts = np.array(centroids, dtype=float)
    n = pts.shape[0]
    labels = [-1] * n
    next_label = 0
    threshold_px = distance_m / float(max(1e-9, mpp))

    for i in range(n):
        if labels[i] != -1:
            continue
        # BFS/expand cluster
        stack = [i]
        labels[i] = next_label
        while stack:
            u = stack.pop()
            dists = np.linalg.norm(pts - pts[u], axis=1)
            neighbors = np.where(dists <= threshold_px)[0]
            for nb in neighbors:
                if labels[nb] == -1:
                    labels[nb] = next_label
                    stack.append(nb)
        next_label += 1
    return labels

def aggregate(detections, w, h, lat=None, zoom=DEFAULT_ZOOM):
    total_area = 0.0
    total_cap = 0.0
    boxes = []
    masks_list = []
    confs = []

    for det in detections:
        box = det.get("box")
        conf = float(det.get("confidence", 0.0))
        mask = det.get("mask")
        if box: boxes.append(box)
        if mask is not None: masks_list.append(mask)
        if mask is not None:
            area, cap = estimateFromMasks(mask, w, h, lat, zoom)
        else:
            area, cap = estimateFromBoxes([box], w, h, lat, zoom)
        total_area += area
        total_cap += cap
        confs.append(conf)

    avg_conf = float(sum(confs)/len(confs)) if confs else 0.0

    # Panel count estimation: area / avg panel area (rounded)
    panel_count = int(max(0, round(total_area / AVG_PANEL_AREA_SQM)))

    # Clustering (group nearby detections)
    centroids = [ _centroid_from_box(b) for b in boxes ]
    mpp = _effective_mpp(lat if lat is not None else 0.0, zoom, w)
    labels = cluster_centroids(centroids, mpp, distance_m=10.0)
    cluster_map = {}
    for lbl, b, conf in zip(labels, boxes, confs):
        if lbl not in cluster_map:
            cluster_map[lbl] = {"boxes": [], "confs": [], "centroid_px": None}
        cluster_map[lbl]["boxes"].append(b)
        cluster_map[lbl]["confs"].append(conf)
    clusters = []
    for lbl, info in cluster_map.items():
        c_pts = [ _centroid_from_box(bb) for bb in info["boxes"] ]
        cx = sum(p[0] for p in c_pts)/len(c_pts)
        cy = sum(p[1] for p in c_pts)/len(c_pts)
        clusters.append({
            "cluster_id": int(lbl),
            "num_detections": len(info["boxes"]),
            "mean_confidence": float(sum(info["confs"])/len(info["confs"])),
            "centroid_px": [float(cx), float(cy)]
        })

    return {
        "panel_count": panel_count,
        "area_sqm": total_area,
        "capacity_kw": total_cap,
        "confidence": avg_conf,
        "has_solar": panel_count > 0 or total_area > 0.1,
        "boxes": boxes,
        "masks": masks_list,
        "clusters": clusters,
        "mpp": mpp
    }

# ============================================================
# YOLO: load + robust embedding extraction
# ============================================================
def loadYolo(weights: str):
    device = "cuda" if TORCH_OK and torch.cuda.is_available() else "cpu"
    model = YOLO(weights)
    try:
        model.fuse()
    except Exception:
        pass
    return model, device

def inferYOLO_with_embedding(model, device, imagePath, imgsz=1024, conf=DEFAULT_CONF, iou=DEFAULT_IOU):
    """
    Runs YOLO and attempts to extract an image embedding.
    Returns (detections_list, embedding_list)
    """
    # run predict
    try:
        res_list = model.predict(source=imagePath, device=device, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    except Exception:
        res_list = model.predict(imagePath)

    if not res_list:
        return [], []

    res = res_list[0]

    # extract boxes/conf/masks
    try:
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
    except Exception:
        boxes = np.array([])
        confs = np.array([])

    masks = None
    try:
        if getattr(res, "masks", None) is not None and getattr(res.masks, "data", None) is not None:
            masks = res.masks.data.cpu().numpy()
    except Exception:
        masks = None

    dets = []
    for i in range(len(boxes)):
        dets.append({
            "box": tuple(float(x) for x in boxes[i]),
            "confidence": float(confs[i]) if i < len(confs) else 0.0,
            "mask": masks[i] if (masks is not None and i < len(masks)) else None
        })

    # embedding extraction tries (robust, many fallbacks)
    embedding = []
    try:
        # 1) try model.predict ret_layer (newer ultralytics)
        try:
            preds = model.predict(source=imagePath, device=device, imgsz=imgsz, conf=conf, iou=iou, verbose=False, ret_layer=-2)
            if preds and hasattr(preds[0], "feats") and preds[0].feats is not None:
                feats = preds[0].feats
                emb_tensor = feats[-1] if isinstance(feats, (list, tuple)) else feats
                if hasattr(emb_tensor, "cpu"):
                    embedding = emb_tensor.cpu().numpy().flatten().tolist()
            elif preds and hasattr(preds[0], "layer_outputs"):
                lo = preds[0].layer_outputs
                emb_tensor = lo[-1]
                embedding = emb_tensor.cpu().numpy().flatten().tolist()
        except Exception:
            # 2) try reading res.features / res.feats
            if hasattr(res, "features"):
                try:
                    embedding = np.asarray(res.features).flatten().tolist()
                except Exception:
                    pass
            if hasattr(res, "feats"):
                try:
                    f = res.feats
                    t = f[-1] if isinstance(f, (list, tuple)) else f
                    embedding = t.cpu().numpy().flatten().tolist()
                except Exception:
                    pass

        # 3) fallback: try model.model forward
        if not embedding:
            try:
                import torch
                if hasattr(model, "model"):
                    img = Image.open(imagePath).convert("RGB")
                    arr = np.array(img.resize((imgsz, imgsz))).astype(np.float32) / 255.0
                    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(next(model.model.parameters()).device)
                    # try last-but-one submodule
                    try:
                        sub = model.model[-2]
                        with torch.no_grad():
                            out = sub(tensor)
                        if hasattr(out, "cpu"):
                            embedding = out.cpu().numpy().flatten().tolist()
                    except Exception:
                        with torch.no_grad():
                            out_full = model.model(tensor)
                        t = out_full[-1] if isinstance(out_full, (list, tuple)) else out_full
                        if hasattr(t, "cpu"):
                            embedding = t.cpu().numpy().flatten().tolist()
            except Exception:
                pass
    except Exception:
        embedding = []

    # normalize & ensure list of floats
    if not isinstance(embedding, list):
        try:
            embedding = list(map(float, np.asarray(embedding).flatten().tolist()))
        except Exception:
            embedding = []

    return dets, embedding

# ============================================================
# PROCESSING
# ============================================================
def processImage(imagePath, weightsPath, inferModel,
                 lat=None, lon=None, zoom=DEFAULT_ZOOM, imgsz=DEFAULT_IMG_SIZE, conf=DEFAULT_CONF,
                 crop_size=640):
    model, device = loadYolo(weightsPath)
    dets, embedding = inferYOLO_with_embedding(model, device, imagePath, imgsz=imgsz, conf=conf, iou=DEFAULT_IOU)

    try:
        with Image.open(imagePath) as im:
            w, h = im.size
    except Exception:
        w, h = imgsz, imgsz

    agg = aggregate(dets, w, h, lat, zoom)

    # Build detData (keys match inference expectation)
    detData = {
        "has_solar": bool(agg["has_solar"]),
        "confidence": float(agg["confidence"]),
        "panel_count": int(agg["panel_count"]),
        "area_sqm": float(agg["area_sqm"]),
        "capacity_kw": float(agg["capacity_kw"]),
        "boxes": agg["boxes"],
        "masks": agg["masks"],
        "embedding": embedding,
        "clusters": agg.get("clusters", []),
        "mpp": agg.get("mpp", None)
    }

    # note: warn if embedding empty
    if not embedding:
        print("[run_model] WARN: embedding empty for image:", imagePath, file=sys.stderr)

    # QC NOTES
    try:
        qc_notes = inferModel.generateQcNotes(detData)
    except Exception as e:
        print("[run_model] QC model failed, falling back:", e, file=sys.stderr)
        qc_notes = generateQcNotes(detData)

    qc_status = "verified" if detData["has_solar"] else "no_solar_detected"

    # Overlay
    overlay_path = str(AUDIT_DIR / (Path(imagePath).stem + "_overlay.png"))
    try:
        saveOverlay(imagePath, agg["masks"], agg["boxes"], overlay_path)
    except Exception:
        overlay_path = ""

    # Append training (includes embedding)
    summary = {
        "confidence": detData["confidence"],
        "panel_count": detData["panel_count"],
        "area_sqm": detData["area_sqm"],
        "capacity_kw": detData["capacity_kw"],
        "has_solar": detData["has_solar"],
        "qc_notes": qc_notes,
        "embedding": embedding
    }
    appended = appendTraining(summary, imagePath)

    # Format bbox_or_mask: prefer mask meta, fallback to boxes
    if agg["masks"]:
        bbox_or_mask = {"masks_available": True, "num_masks": len(agg["masks"])}
    else:
        bbox_or_mask = agg["boxes"]

    sample_id = int(time.time() * 1000) % (10**12)

    out = {
        "sample_id": sample_id,
        "lat": float(lat) if lat is not None else None,
        "lon": float(lon) if lon is not None else None,
        "has_solar": detData["has_solar"],
        "confidence": detData["confidence"],
        "panel_count_Est": detData["panel_count"],    # Judges' field
        "pv_area_sqm_est": detData["area_sqm"],
        "capacity_kw_est": detData["capacity_kw"],
        "qc_status": qc_status,
        "qc_notes": qc_notes,
        "bbox_or_mask": bbox_or_mask,
        "clusters": agg.get("clusters", []),
        "mpp": agg.get("mpp", None),
        "image_metadata": {
            "source": Path(imagePath).name,
            "audit_overlay_path": overlay_path,
            "capture_date": None
        },
        "appended_to_training": appended,
        "timestamp": nowStr()
    }

    # Print single JSON object (judge expects object starting with { )
    print(json.dumps(out))
    return out

# ============================================================
# CLI ENTRY
# ============================================================
def main():
    if len(sys.argv) < 3:
        print("Usage: python run_model.py <image> <weights> [--lat <lat> --lon <lon> --zoom <z> --crop-size <px>]", file=sys.stderr)
        sys.exit(1)

    imagePath = sys.argv[1]
    weightsPath = sys.argv[2]

    # optional args
    lat = None
    lon = None
    zoom = DEFAULT_ZOOM
    crop_size = 640
    i = 3
    while i < len(sys.argv):
        a = sys.argv[i]
        if a in ("--lat", "-lat"):
            lat = float(sys.argv[i+1]); i += 2; continue
        if a in ("--lon", "-lon"):
            lon = float(sys.argv[i+1]); i += 2; continue
        if a in ("--zoom", "-z"):
            zoom = int(sys.argv[i+1]); i += 2; continue
        if a in ("--crop-size",):
            crop_size = int(sys.argv[i+1]); i += 2; continue
        i += 1

    inferModel = Model(modelPath=MODEL_PICKLE_PATH, trainingJsonPath=TRAINING_JSON_PATH)
    processImage(imagePath, weightsPath, inferModel, lat=lat, lon=lon, zoom=zoom, imgsz=DEFAULT_IMG_SIZE, conf=DEFAULT_CONF, crop_size=crop_size)

if __name__ == "__main__":
    main()
