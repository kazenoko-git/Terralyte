#!/usr/bin/env python3
"""
run_model.py — Patched full inference pipeline.

Features:
- Accepts image input as: path to .b64 file, path to image, or raw base64 string.
- Supports optional sidecar metadata file: <input>.meta.json (lat/lon/zoom/capture_date)
- Writes audit overlay to PROJECT_ROOT/tmp_input_overlay.png
- Writes detection JSON to PROJECT_ROOT/detections/detection_<sample_id>.json
- Keeps judge fields (panel_count_Est) and adds image_metadata.capture_date
- Corrects meters-per-pixel math (use metersPerPixel(lat, zoom))
- Draws black label rectangles AND writes confidence text inside them
- Appends training data safely (dedupe)
"""

import os
os.environ["YOLO_VERBOSE"] = "False"
os.environ["ULTRALYTICS_HUB"] = "False"
os.environ["ULTRALYTICS_LOGGING"] = "False"

import warnings
warnings.filterwarnings("ignore")

import sys
import json
import math
import time
import hashlib
import base64
import io
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np

def silent(*a, **k): pass

# Try imports that may fail gracefully
try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# Local QC model (your provided file)
try:
    from inference import Model, generateQcNotes
except Exception:
    # If inference import fails, define fallback generateQcNotes
    def generateQcNotes(d):
        return ["QC unavailable"]
    Model = None

# ---------------------------
# PROJECT PATHS — DERIVE PROJECT_ROOT
# ---------------------------
SCRIPT_PATH = Path(__file__).resolve()
# If run_model.py is inside repo root: parent is project root.
# If it's in a subfolder, parent.parent may be needed; prefer parent (safe).
PROJECT_ROOT = SCRIPT_PATH.parent  # keep simple and safe; adjust if your layout differs
# If you want explicit path, uncomment below:
# PROJECT_ROOT = Path("/Users/ivansamuel/PycharmProjects/Terralyte")

AUDIT_DIR = PROJECT_ROOT / "audit_overlays"
DETECTIONS_DIR = PROJECT_ROOT / "detections"
OVERLAY_SAVE_PATH = PROJECT_ROOT / "tmp_input_overlay.png"
TRAINING_JSON_PATH = PROJECT_ROOT / "qc_training_data.json"
MODEL_PATH = PROJECT_ROOT / "solarQcModel.pkl"

# Ensure directories exist
for p in (AUDIT_DIR, DETECTIONS_DIR):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[run_model] WARN: could not create dir {p}: {e}", file=sys.stderr)

# log resolved roots for debugging
print(f"[run_model] PROJECT_ROOT = {PROJECT_ROOT}", file=sys.stderr)
print(f"[run_model] DETECTIONS_DIR = {DETECTIONS_DIR}", file=sys.stderr)
print(f"[run_model] AUDIT_DIR = {AUDIT_DIR}", file=sys.stderr)

# ---------------------------
# Config
# ---------------------------
DEFAULT_IMG_SIZE = 1024
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_ZOOM = 18
TILE_SIZE = 256

PANEL_POWER_DENSITY = 180.0   # W/m²
AVG_PANEL_AREA_SQM = 1.7      # m² per panel estimate

# ---------------------------
# Helpers
# ---------------------------
def nowStr():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def shaDetection(imagePath: str, summary: Dict):
    s = f"{imagePath}|{summary.get('panel_count',0)}|{summary.get('area_sqm',0)}|{summary.get('capacity_kw',0)}"
    return hashlib.sha256(s.encode()).hexdigest()

def metersPerPixel(lat_deg: float, zoom: int):
    # meters per pixel at given zoom using Web Mercator
    return 156543.03392 * math.cos(math.radians(lat_deg)) / (2 ** zoom)

# ---------------------------
# Area / Capacity Estimators (FIXED)
# ---------------------------
def estimateFromMasks(masks, width, height, lat=None, zoom=DEFAULT_ZOOM):
    """
    masks: array-like of binary masks (full image size)
    width/height: pixel dims of the image used by YOLO (cropped image)
    lat/zoom: required to convert pixel -> meters. If missing, approximate with defaults.
    """
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
    if zoom is None:
        zoom = DEFAULT_ZOOM

    mpp = metersPerPixel(lat, zoom)
    sqm = pixel_count * (mpp ** 2)
    # conservative capacity factor
    kw = (sqm * PANEL_POWER_DENSITY / 1000.0) * 0.80
    return float(sqm), float(min(max(kw, 0.0), 20000.0))

def estimateFromBoxes(boxes, width, height, lat=None, zoom=DEFAULT_ZOOM):
    """
    boxes are in the same pixel space as the image.
    width/height: image dimensions
    lat/zoom: necessary for accurate mpp calculation; fallback otherwise.
    """
    if not boxes:
        return 0.0, 0.0

    if lat is None:
        lat = 0.0
    if zoom is None:
        zoom = DEFAULT_ZOOM

    mpp = metersPerPixel(lat, zoom)
    total_pixels = 0.0
    for (x1, y1, x2, y2) in boxes:
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        total_pixels += w * h

    sqm = total_pixels * (mpp ** 2)
    kw = (sqm * PANEL_POWER_DENSITY / 1000.0) * 0.75
    return float(sqm), float(min(max(kw, 0.0), 20000.0))

# ---------------------------
# Overlay drawing (keeps black rectangle but writes confidence)
# ---------------------------
def saveOverlay(imagePathOrPIL, masks, boxes, outPath, confidences=None, alpha=0.35):
    """
    imagePathOrPIL: either path string or PIL.Image
    masks: list/array of masks
    boxes: list of boxes [(x1,y1,x2,y2), ...]
    confidences: optional list of confidences aligned with boxes
    """
    try:
        if isinstance(imagePathOrPIL, str):
            img = Image.open(imagePathOrPIL).convert("RGBA")
        else:
            img = imagePathOrPIL.convert("RGBA")

    except Exception as e:
        print("[run_model] overlay: failed to open base image:", e, file=sys.stderr)
        return None

    canvas = Image.new("RGBA", img.size)
    draw = ImageDraw.Draw(canvas, "RGBA")

    # paste masks (prefer full-size masks)
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

    # Draw bounding boxes and confidence label (black rectangle + text)
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        # box
        draw.rectangle([x1,y1,x2,y2], outline=(255,255,0,255), width=2)
        # black label background (keep it as user wanted)
        label_w = 80
        label_h = 16
        lx1 = x1
        ly1 = max(0, y1 - label_h)
        lx2 = lx1 + label_w
        ly2 = y1
        draw.rectangle([lx1, ly1, lx2, ly2], fill=(0,0,0,200))
        # label text: confidence (if provided) else index
        label_txt = f"{(confidences[idx]*100):.0f}%" if confidences and idx < len(confidences) else f"#{idx+1}"
        try:
            # default font; PIL will fallback if not found
            draw.text((lx1+4, ly1+1), label_txt, fill=(255,255,255,255))
        except Exception:
            # if text fails, ignore (label rectangle still present)
            pass

    out = Image.alpha_composite(img, canvas)
    try:
        out = out.filter(ImageFilter.SHARPEN)
    except Exception:
        pass

    try:
        out.save(outPath, format="PNG")
        return str(outPath)
    except Exception as e:
        print("[run_model] overlay: save failed:", e, file=sys.stderr)
        # fallback to saving without alpha
        try:
            out.convert("RGB").save(outPath, format="PNG")
            return str(outPath)
        except Exception:
            return None

# ---------------------------
# Training JSON append (safe)
# ---------------------------
def appendTraining(summary: Dict, imagePath: str):
    TRAINING_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    if TRAINING_JSON_PATH.exists():
        try:
            data = json.load(open(TRAINING_JSON_PATH))
        except Exception:
            data = []
    else:
        data = []

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
    try:
        json.dump(data, open(TRAINING_JSON_PATH, "w"), indent=2)
    except Exception as e:
        print("[run_model] appendTraining: write failed:", e, file=sys.stderr)
    return True

# ---------------------------
# Clustering helpers
# ---------------------------
def _centroid_from_box(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def cluster_centroids(centroids, mpp, distance_m=10.0):
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
        box = det.get("box") or det.get("bbox") or det.get("box_xyxy")
        conf = float(det.get("confidence", 0.0))
        mask = det.get("mask")
        if box:
            try:
                box_t = tuple(float(x) for x in box)
                boxes.append(box_t)
            except Exception:
                pass
        if mask is not None:
            masks_list.append(mask)
            area, cap = estimateFromMasks(mask, w, h, lat, zoom)
        else:
            area, cap = estimateFromBoxes([box], w, h, lat, zoom)
        total_area += area
        total_cap += cap
        confs.append(conf)

    avg_conf = float(sum(confs)/len(confs)) if confs else 0.0

    panel_count = int(max(0, round(total_area / AVG_PANEL_AREA_SQM)))

    centroids = [ _centroid_from_box(b) for b in boxes ]
    mpp = metersPerPixel(lat if lat is not None else 0.0, zoom if zoom is not None else DEFAULT_ZOOM)
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

# ---------------------------
# YOLO load + embedding
# ---------------------------
def loadYolo(weights: str):
    device = "cuda" if TORCH_OK and torch.cuda.is_available() else "cpu"
    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO not available in this environment")
    model = YOLO(str(weights))
    try:
        model.fuse()
    except Exception:
        pass
    return model, device

def inferYOLO_with_embedding(model, device, imagePath_or_bytes, imgsz=1024, conf=DEFAULT_CONF, iou=DEFAULT_IOU):
    """
    Accepts either:
      - path to image file
      - bytes
      - PIL.Image (will be saved to temp file)
    Returns detections list and embedding list.
    """
    temp_file = None
    input_for_model = None

    try:
        if isinstance(imagePath_orBytes := imagePath_or_bytes, (bytes, bytearray)):
            fd, tmp = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            with open(tmp, "wb") as f:
                f.write(imagePath_orBytes)
            temp_file = tmp
            input_for_model = tmp
        elif isinstance(imagePath_or_bytes, Image.Image):
            fd, tmp = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            imagePath_or_bytes.save(tmp)
            temp_file = tmp
            input_for_model = tmp
        else:
            input_for_model = str(imagePath_or_bytes)
    except Exception:
        input_for_model = str(imagePath_or_bytes)

    # run predict
    try:
        res_list = model.predict(source=input_for_model, device=device, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    except Exception:
        res_list = model.predict(input_for_model)

    if not res_list:
        if temp_file:
            try: os.remove(temp_file)
            except: pass
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

    # embedding extraction attempts
    embedding = []
    try:
        try:
            preds = model.predict(source=input_for_model, device=device, imgsz=imgsz, conf=conf, iou=iou, verbose=False, ret_layer=-2)
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

        if not embedding:
            if hasattr(model, "model") and TORCH_OK:
                img_pil = Image.open(input_for_model).convert("RGB")
                arr = np.array(img_pil.resize((imgsz, imgsz))).astype(np.float32) / 255.0
                import torch
                tensor = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(next(model.model.parameters()).device)
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
        embedding = []

    try:
        if not isinstance(embedding, list):
            embedding = list(map(float, np.asarray(embedding).flatten().tolist()))
    except Exception:
        embedding = []

    if temp_file:
        try: os.remove(temp_file)
        except: pass

    return dets, embedding

# ---------------------------
# Input decoding helpers
# ---------------------------
def is_probable_raw_base64(s: str) -> bool:
    s2 = s.strip()
    # quick heuristic: length & chars
    if len(s2) < 200:
        return False
    ok_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r")
    return all(c in ok_chars for c in s2[:200])

def load_image_input(arg: str) -> Tuple[str, Optional[Image.Image], Optional[str]]:
    """
    Returns tuple: (resolved_input, PIL.Image or None, meta_sidecar_path or None)
    resolved_input:
      - path to a temporary PNG file (if we had base64)
      - original file path (if provided)
    Also returns PIL.Image if convenient (to avoid re-opening later).
    """
    # If arg is a path to a .b64 file (what Tauri writes), read it
    p = Path(arg)
    if p.suffix.lower() == ".b64" and p.exists():
        data = p.read_text().strip()
        # if file contains data URL prefix, strip
        if data.startswith("data:"):
            data = data.split(",",1)[1]
        try:
            img_bytes = base64.b64decode(data)
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
            fd, tmp = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            pil.save(tmp, format="PNG")
            return tmp, pil, str(p.with_suffix(".meta.json"))
        except Exception as e:
            raise RuntimeError(f"Failed to decode .b64 file {arg}: {e}")

    # If arg looks like raw base64, decode and save to temp png
    if is_probable_raw_base64(arg):
        data = arg
        try:
            if data.startswith("data:"):
                data = data.split(",",1)[1]
            img_bytes = base64.b64decode(data)
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
            fd, tmp = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            pil.save(tmp, format="PNG")
            return tmp, pil, None
        except Exception as e:
            raise RuntimeError(f"Failed to decode base64 input: {e}")

    # Otherwise assume arg is a filesystem path to an image
    if p.exists():
        try:
            pil = Image.open(str(p)).convert("RGBA")
            return str(p), pil, str(p.with_suffix(".meta.json"))
        except Exception as e:
            # can't open image
            raise RuntimeError(f"Failed to open image path {arg}: {e}")

    # fallback: assume it's a raw string but not base64 -> error
    raise RuntimeError(f"Unable to interpret input argument: {arg}")

# ---------------------------
# Main processing
# ---------------------------
def processImage(imageInputArg, weightsPath, inferModel,
                 lat=None, lon=None, zoom=None, imgsz=DEFAULT_IMG_SIZE, conf=DEFAULT_CONF,
                 crop_size=640, capture_date: Optional[str]=None):
    """
    imageInputArg: path to .b64 or path to image or raw base64 string
    weightsPath: path to YOLO weights
    inferModel: Model instance for QC notes (can be None)
    lat/lon/zoom: optional (can be read from sidecar meta)
    """
    resolved_path = None
    pil_img = None
    meta_sidecar = None

    try:
        resolved_path, pil_img, meta_sidecar = load_image_input(imageInputArg)
        print(f"[run_model] loaded image input -> {resolved_path}", file=sys.stderr)
    except Exception as e:
        raise RuntimeError(f"Failed to load image input: {e}")

    # If there's a sidecar metadata file, load lat/lon/zoom/capture_date from it if not provided
    if meta_sidecar:
        try:
            meta_p = Path(meta_sidecar)
            if meta_p.exists():
                meta = json.load(open(meta_p, "r"))
                lat = lat if lat is not None else meta.get("lat", lat)
                lon = lon if lon is not None else meta.get("lon", lon)
                zoom = zoom if zoom is not None else meta.get("zoom", zoom)
                capture_date = capture_date if capture_date is not None else meta.get("capture_date", capture_date)
                print(f"[run_model] loaded sidecar meta {meta_sidecar}: {meta}", file=sys.stderr)
        except Exception as e:
            print(f"[run_model] warn: failed reading sidecar meta {meta_sidecar}: {e}", file=sys.stderr)

    # If still missing lat/zoom, set defaults but warn
    if lat is None:
        lat = 0.0
        print("[run_model] WARN: lat missing, using 0.0", file=sys.stderr)
    if lon is None:
        lon = 0.0
        print("[run_model] WARN: lon missing, using 0.0", file=sys.stderr)
    if zoom is None:
        zoom = DEFAULT_ZOOM
        print(f"[run_model] WARN: zoom missing, using default {DEFAULT_ZOOM}", file=sys.stderr)

    # Load YOLO
    model, device = loadYolo(weightsPath)

    # Run inference
    dets, embedding = inferYOLO_with_embedding(model, device, resolved_path, imgsz=imgsz, conf=conf, iou=DEFAULT_IOU)

    # Image dims
    try:
        w, h = pil_img.size
    except Exception:
        w, h = imgsz, imgsz

    # Aggregate
    agg = aggregate(dets, w, h, lat, zoom)

    # Build detData
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

    if not embedding:
        print("[run_model] WARN: embedding empty for image", file=sys.stderr)

    # QC notes
    try:
        if inferModel is not None:
            qc_notes = inferModel.generateQcNotes(detData)
        else:
            qc_notes = generateQcNotes(detData)
    except Exception as e:
        print("[run_model] QC model failed, falling back:", e, file=sys.stderr)
        qc_notes = generateQcNotes(detData)

    qc_status = "verifiable" if detData["has_solar"] else "no_solar_detected"

    # Save overlay (put confidence text inside black rectangles)
    overlay_path = str(OVERLAY_SAVE_PATH)
    try:
        confidences = [float(d.get("confidence", 0.0)) for d in dets] if dets else None
        saved = saveOverlay(pil_img, agg["masks"], agg["boxes"], OVERLAY_SAVE_PATH, confidences=confidences)
        if not saved:
            overlay_path = ""
    except Exception as e:
        overlay_path = ""
        print("[run_model] overlay save failed:", e, file=sys.stderr)

    # Append training
    summary = {
        "confidence": detData["confidence"],
        "panel_count": detData["panel_count"],
        "area_sqm": detData["area_sqm"],
        "capacity_kw": detData["capacity_kw"],
        "has_solar": detData["has_solar"],
        "qc_notes": qc_notes,
        "embedding": embedding
    }
    appended = appendTraining(summary, resolved_path)

    # bbox_or_mask: prefer boxes (list) if masks absent, else compact mask info
    if agg["masks"]:
        bbox_or_mask = {"masks_present": True, "num_masks": len(agg["masks"])}
    else:
        bbox_or_mask = agg["boxes"]

    # sample_id deterministic-ish
    sample_id = int(time.time() * 1000) % (10**12)

    cap_date = capture_date if capture_date else nowStr()

    out = {
        "sample_id": sample_id,
        "lat": float(lat) if lat is not None else None,
        "lon": float(lon) if lon is not None else None,
        "has_solar": detData["has_solar"],
        "confidence": detData["confidence"],
        "panel_count_Est": detData["panel_count"],
        "pv_area_sqm_est": detData["area_sqm"],
        "capacity_kw_est": detData["capacity_kw"],
        "qc_status": qc_status,
        "qc_notes": qc_notes,
        "bbox_or_mask": bbox_or_mask,
        "clusters": agg.get("clusters", []),
        "mpp": agg.get("mpp", None),
        "audit_overlay_path": overlay_path,
        "image_metadata": {
            "source": Path(resolved_path).name if resolved_path else None,
            "capture_date": cap_date,
            "audit_overlay_path": overlay_path
        },
        "appended_to_training": appended,
        "timestamp": nowStr()
    }

    # Save detection JSON
    try:
        out_path = DETECTIONS_DIR / f"detection_{sample_id}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[run_model] saved detection JSON -> {out_path}", file=sys.stderr)
    except Exception as e:
        print("[run_model] failed to save detection json:", e, file=sys.stderr)

    # Print JSON (single object)
    print(json.dumps(out))

    # cleanup temp if used
    try:
        if resolved_path and resolved_path.endswith(".png") and Path(resolved_path).name.startswith("tmp"):
            try:
                Path(resolved_path).unlink()
            except Exception:
                pass
    except Exception:
        pass

    return out

# ---------------------------
# CLI ENTRY
# ---------------------------
def main():
    """
    Usage:
      python run_model.py <imageInput> <weights> [--lat <lat> --lon <lon> --zoom <z> --capture-date YYYY-MM-DD]
    imageInput: path to .b64, path to .png, or raw base64 string
    """
    if len(sys.argv) < 3:
        print("Usage: python run_model.py <imageInput> <weights> [--lat <lat> --lon <lon> --zoom <z> --capture-date YYYY-MM-DD]", file=sys.stderr)
        sys.exit(1)

    raw_input = sys.argv[1]
    weightsPath = sys.argv[2]

    lat = None
    lon = None
    zoom = None
    capture_date = None
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
        if a in ("--capture-date",):
            capture_date = sys.argv[i+1]; i += 2; continue
        if a in ("--crop-size",):
            crop_size = int(sys.argv[i+1]); i += 2; continue
        i += 1

    # initialize QC model if available
    inferModel = None
    try:
        inferModel = Model(modelPath=MODEL_PATH, trainingJsonPath=str(TRAINING_JSON_PATH))
    except Exception as e:
        print("[run_model] WARN: failed to init QC Model:", e, file=sys.stderr)

    try:
        processImage(raw_input, weightsPath, inferModel, lat=lat, lon=lon, zoom=zoom, imgsz=DEFAULT_IMG_SIZE, conf=DEFAULT_CONF, crop_size=crop_size, capture_date=capture_date)
    except Exception as e:
        print("[run_model] ERROR during processing:", e, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
